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

    // ---------------------------------------------------------- 权重预处理
    // JSON 里权重存成 [输入维][输出维]（numpy 的行主序）。推理时需要的是
    // 「第 k 个输出神经元对全部输入的点积」，也就是要沿着输入维连续取值。
    //
    // 这里做两件事：
    //   1) 转置成 [输出维][输入维]，并**展平成一个连续的 Float64Array**。
    //      展平后每次取值都是 TypedArray 的下标访问，比先取 rows[k]（对象属性）、
    //      再取 row[m]（又一层）少一次指针跳转，也更容易被 JIT 优化成线性内存访问。
    //   2) 预分配两块缓冲区轮流复用。原来每层都 new 一次 Float64Array，
    //      高频调用时会把时间耗在分配与 GC 上。
    //
    // 注意：转置改变的只是存储布局，累加顺序没有变，因此计算结果与 Python
    // 训练端保持逐位一致（由 verify_surrogate.js 守门）。
    this.Wflat = [];
    this.b = [];
    this.inDims = [];
    this.outDims = [];
    var maxDim = 1;
    for (var i = 0; i < this.nLayers; i++) {
      var W = payload.network.weights[i];
      var outDim = this.layerSizes[i + 1];
      var inDim = this.layerSizes[i];
      var flat = new Float64Array(outDim * inDim);
      for (var k = 0; k < outDim; k++) {
        var base = k * inDim;
        var col = W;
        for (var m = 0; m < inDim; m++) flat[base + m] = col[m][k];
      }
      this.Wflat.push(flat);
      this.b.push(Float64Array.from(payload.network.biases[i]));
      this.inDims.push(inDim);
      this.outDims.push(outDim);
      if (outDim > maxDim) maxDim = outDim;
      if (inDim > maxDim) maxDim = inDim;
    }

    // 两块乒乓缓冲区，避免 predictVector 内部反复分配
    this._bufA = new Float64Array(maxDim);
    this._bufB = new Float64Array(maxDim);
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

  /**
   * 最底层接口：输入原始物理量向量，输出原始物理量向量。
   *
   * 性能要点（这段代码是全局被调用最密集的地方）：
   *   - 输入归一化与逐层前向都在复用的缓冲区里完成，全程只有最后一次
   *     输出需要分配数组；
   *   - 层间用双缓冲交替，省掉每层一次 new Float64Array；
   *   - 内层点积把长度与权重起始偏移提到循环外，避免重复的属性查找。
   *
   * 返回的 Float64Array 是新分配的对象（调用方可能长期持有），
   * 但每层缓冲区都是复用的，因此热路径上不再有中间分配。
   */
  Surrogate.prototype.predictVector = function (xRaw) {
    var xm = this.norm.x_mean, xs = this.norm.x_std;
    var a = this._bufA, b = this._bufB;
    var n = this.layerSizes[0];
    for (var j = 0; j < n; j++) a[j] = (xRaw[j] - xm[j]) / xs[j];

    var nLayers = this.nLayers;
    for (var i = 0; i < nLayers; i++) {
      var W = this.Wflat[i], bias = this.b[i];
      var outDim = this.outDims[i], inDim = this.inDims[i];
      var isHidden = i < nLayers - 1;
      for (var k = 0; k < outDim; k++) {
        var base = k * inDim;
        var s = bias[k];
        for (var m = 0; m < inDim; m++) s += a[m] * W[base + m];
        b[k] = (isHidden && s < 0) ? 0 : s;   // 隐层 ReLU，输出层线性
      }
      var t = a; a = b; b = t;               // 交换缓冲区，a 持有本层输出
    }

    var ym = this.norm.y_mean, ys = this.norm.y_std;
    var nOut = this.layerSizes[nLayers];
    // 输出仍返回新数组：调用方可能长期持有它，复用会破坏语义。
    // 这个 6 元素的小分配相对矩阵乘的开销可以忽略。
    var out = new Float64Array(nOut);
    if (this.logSpace) {
      for (var q = 0; q < nOut; q++) out[q] = Math.exp(a[q] * ys[q] + ym[q]);
    } else {
      for (var q2 = 0; q2 < nOut; q2++) out[q2] = a[q2] * ys[q2] + ym[q2];
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
