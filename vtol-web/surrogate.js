/**
 * 代理模型的浏览器端推理 —— 零依赖，与 numpy 训练端数值一致。
 *
 * 加载方式：页面先引入 surrogate-data.js（它会把权重挂在
 * window.VTOL_SURROGATE_DATA 上），再用这里的 Surrogate 类做推理。
 *
 * 为什么不用 fetch 读 JSON：浏览器以 file:// 协议打开页面时，
 * fetch 本地文件会被 CORS 拦截，而 <script src> 不受影响。
 * 所以权重导出成了 .js 而不是纯 .json。
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.VTOL_SURROGATE = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /**
   * @param {Object} payload 由 surrogate/train.py 导出的权重包
   */
  function Surrogate(payload) {
    if (!payload || !payload.network) {
      throw new Error('代理模型权重格式不正确');
    }
    this.payload = payload;
    this.meta = payload.meta;
    this.layerSizes = payload.network.layer_sizes;
    this.nLayers = this.layerSizes.length - 1;
    this.norm = payload.normalization;
    this.logSpace = !!payload.normalization.log_space;

    this.paramNames = payload.params.map(function (p) { return p.name; });
    this.paramMeta = payload.params;
    this.outputNames = payload.outputs.map(function (o) { return o.name; });
    this.outputMeta = payload.outputs;

    // 预转置权重：JSON 里是 [输入维][输出维]，但推理时的内层循环
    // 沿着输出维跑，转置成 [输出维][输入维] 后内存连续，快很多。
    this.Wt = [];
    this.b = [];
    for (var i = 0; i < this.nLayers; i++) {
      var W = payload.network.weights[i];
      var outDim = this.layerSizes[i + 1];
      var inDim = this.layerSizes[i];
      var rows = new Array(outDim);
      for (var k = 0; k < outDim; k++) {
        var row = new Float64Array(inDim);
        for (var m = 0; m < inDim; m++) row[m] = W[m][k];
        rows[k] = row;
      }
      this.Wt.push(rows);
      this.b.push(Float64Array.from(payload.network.biases[i]));
    }
  }

  /** 参数是否落在训练区间内。返回越界参数的中文标签列表。 */
  Surrogate.prototype.outOfRange = function (xRaw) {
    var bad = [];
    for (var j = 0; j < this.paramMeta.length; j++) {
      var p = this.paramMeta[j];
      if (xRaw[j] < p.low - 1e-9 || xRaw[j] > p.high + 1e-9) {
        bad.push(p.label);
      }
    }
    return bad;
  };

  /** 最底层接口：输入原始物理量向量，输出原始物理量向量。 */
  Surrogate.prototype.predictVector = function (xRaw) {
    var xm = this.norm.x_mean, xs = this.norm.x_std;
    var a = new Float64Array(xRaw.length);
    for (var j = 0; j < xRaw.length; j++) a[j] = (xRaw[j] - xm[j]) / xs[j];

    for (var i = 0; i < this.nLayers; i++) {
      var rows = this.Wt[i], bias = this.b[i];
      var outDim = rows.length;
      var z = new Float64Array(outDim);
      var isHidden = i < this.nLayers - 1;
      for (var k = 0; k < outDim; k++) {
        var row = rows[k], s = bias[k];
        for (var m = 0; m < a.length; m++) s += a[m] * row[m];
        z[k] = (isHidden && s < 0) ? 0 : s;   // 隐层 ReLU，输出层线性
      }
      a = z;
    }

    var ym = this.norm.y_mean, ys = this.norm.y_std;
    var out = new Float64Array(a.length);
    for (var q = 0; q < a.length; q++) {
      var v = a[q] * ys[q] + ym[q];
      out[q] = this.logSpace ? Math.exp(v) : v;
    }
    return out;
  };

  /** 按参数名取值：传入 {rotor_radius: 0.9, ...}，返回 {mtow: ..., ...}。 */
  Surrogate.prototype.predict = function (paramObject) {
    var x = new Float64Array(this.paramNames.length);
    for (var j = 0; j < this.paramNames.length; j++) {
      var v = paramObject[this.paramNames[j]];
      x[j] = (typeof v === 'number') ? v : 0;
    }
    var y = this.predictVector(x);
    var out = {};
    for (var k = 0; k < this.outputNames.length; k++) out[this.outputNames[k]] = y[k];
    return out;
  };

  /** 从页面上的配置对象（camelCase 字段）提取代理模型需要的输入向量。 */
  Surrogate.prototype.configToVector = function (cfg) {
    // 代理模型的字段名与 Python 配置一致（snake_case），页面用 camelCase，
    // 这里做一次映射。
    var map = {
      rotor_radius: cfg.rotorRadius,
      n_rotor: cfg.nRotor,
      fm: cfg.fm,
      eta_motor: cfg.etaMotor,
      wing_area: cfg.wingArea,
      aspect_ratio: cfg.aspectRatio,
      structure_fraction: cfg.structureFraction,
      battery_specific_energy: cfg.batterySpecificEnergy,
      cruise_speed: cfg.cruiseSpeed,
      range_km: cfg.rangeKm
    };
    var x = new Float64Array(this.paramNames.length);
    for (var j = 0; j < this.paramNames.length; j++) {
      var v = map[this.paramNames[j]];
      x[j] = (typeof v === 'number') ? v : 0;
    }
    return x;
  };

  /** 整体质量指标（训练时在测试集上测得的）。 */
  Surrogate.prototype.metrics = function () {
    var r2 = [], mape = [];
    for (var i = 0; i < this.outputMeta.length; i++) {
      r2.push(this.outputMeta[i].r2);
      mape.push(this.outputMeta[i].mape_pct);
    }
    var meanR2 = r2.reduce(function (a, b) { return a + b; }, 0) / (r2.length || 1);
    return { r2: r2, mape_pct: mape, mean_r2: meanR2 };
  };

  /** 从页面全局变量构造实例。 */
  function fromGlobal(globalName) {
    var key = globalName || 'VTOL_SURROGATE_DATA';
    var data = (typeof window !== 'undefined') ? window[key] : undefined;
    if (!data) {
      throw new Error('未找到代理模型权重，请确认已引入 surrogate-data.js');
    }
    return new Surrogate(data);
  }

  return {
    Surrogate: Surrogate,
    fromGlobal: fromGlobal
  };
});
