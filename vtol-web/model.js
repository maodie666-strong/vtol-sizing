/**
 * vtol_sizing 的 JavaScript 移植版 —— 与 Python 内核同模型、同参数。
 *
 * 用途：为 Web 界面提供零依赖的物理计算。浏览器与 Node 均可加载（UMD）。
 * 模型细节见 README.md 与 vtol_sizing/ 下的 Python 源码。
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.VTOL = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------------------------------------------------------------- 常量
  var G0 = 9.80665;
  var T0 = 288.15;
  var P0 = 101325.0;
  var LAPSE = 0.0065;
  var R_AIR = 287.05287;

  // ------------------------------------------------------------ ISA 大气
  function temperature(h) {
    return T0 - LAPSE * h;
  }

  function pressure(h) {
    return P0 * Math.pow(1.0 - LAPSE * h / T0, G0 / (R_AIR * LAPSE));
  }

  function density(h) {
    return pressure(h) / (R_AIR * temperature(h));
  }

  // ---------------------------------------------------------------- 旋翼
  function diskArea(nRotor, radius) {
    return nRotor * Math.PI * radius * radius;
  }

  function hoverInducedVelocity(thrust, rho, area) {
    return Math.sqrt(thrust / (2.0 * rho * area));
  }

  function hoverShaftPower(thrust, rho, area, fm) {
    return thrust * hoverInducedVelocity(thrust, rho, area) / fm;
  }

  function axialClimbShaftPower(thrust, rho, area, fm, climbRate) {
    var vh = hoverInducedVelocity(thrust, rho, area);
    var vi = (-climbRate + Math.sqrt(climbRate * climbRate + 4.0 * vh * vh)) / 2.0;
    return thrust * (climbRate + vi) / fm;
  }

  function toElectricPower(shaftPower, etaMotor, etaTransmission) {
    return shaftPower / (etaMotor * etaTransmission);
  }

  // ---------------------------------------------------------------- 机翼
  function liftCurveSlope(aspectRatio) {
    return 2.0 * Math.PI * aspectRatio / (aspectRatio + 2.0);
  }

  function wingPolar(weight, rho, speed, area, aspectRatio, cd0, oswaldE) {
    var q = 0.5 * rho * speed * speed;
    var cl = weight / (q * area);
    var cd = cd0 + cl * cl / (Math.PI * aspectRatio * oswaldE);
    return {
      cl: cl,
      cd: cd,
      liftToDrag: cd > 0 ? cl / cd : Infinity,
      drag: q * area * cd
    };
  }

  function cruisePower(weight, rho, speed, area, aspectRatio, cd0, oswaldE, etaProp, etaMotor) {
    var polar = wingPolar(weight, rho, speed, area, aspectRatio, cd0, oswaldE);
    var pShaft = polar.drag * speed / etaProp;
    return { power: pShaft / etaMotor, polar: polar };
  }

  // ---------------------------------------------------------------- 能量
  function usableEnergy(requiredEnergy, usableFraction) {
    return requiredEnergy / usableFraction;
  }

  // ------------------------------------------------------------ 默认配置
  function defaultConfig() {
    return {
      // 旋翼系统
      nRotor: 8,
      rotorRadius: 0.90,
      fm: 0.72,
      etaMotor: 0.90,
      etaTransmission: 0.97,
      // 机翼
      wingArea: 6.0,
      aspectRatio: 8.0,
      cd0: 0.028,
      oswaldE: 0.80,
      clMax: 1.50,
      // 重量估算
      structureFraction: 0.30,
      avionicsMass: 25.0,
      payloadMass: 200.0,
      motorSpecificPower: 5000.0,
      rotorSpecificThrust: 250.0,
      powerMargin: 1.15,
      thrustToWeight: 1.30,
      // 电池
      batterySpecificEnergy: 250.0,
      batteryUsableFraction: 0.85,
      // 任务剖面
      hoverTimeS: 60.0,
      climbRate: 3.0,
      climbAltitude: 300.0,
      cruiseSpeed: 55.0,
      cruiseAltitude: 500.0,
      cruisePropEfficiency: 0.78,
      reserveFraction: 0.20,
      rangeKm: 100.0,
      // 求解控制
      mtowGuess: 1000.0,
      relax: 1.5,
      tol: 1e-7,
      maxIter: 400
    };
  }

  // ------------------------------------------------------------ 正向评估
  function evaluate(cfg, mtow) {
    var w = mtow * G0;
    var rhoSL = density(0.0);
    var rhoCruise = density(cfg.cruiseAltitude);
    var area = diskArea(cfg.nRotor, cfg.rotorRadius);

    // 功率
    var pHover = toElectricPower(
      hoverShaftPower(w, rhoSL, area, cfg.fm),
      cfg.etaMotor, cfg.etaTransmission
    );
    var pClimb = toElectricPower(
      axialClimbShaftPower(w, rhoSL, area, cfg.fm, cfg.climbRate),
      cfg.etaMotor, cfg.etaTransmission
    );
    var cruise = cruisePower(
      w, rhoCruise, cfg.cruiseSpeed, cfg.wingArea, cfg.aspectRatio,
      cfg.cd0, cfg.oswaldE, cfg.cruisePropEfficiency, cfg.etaMotor
    );
    var pCruise = cruise.power;
    var polar = cruise.polar;

    // 任务时间
    var tClimb = cfg.climbAltitude / cfg.climbRate;
    var tCruise = cfg.rangeKm * 1000.0 / cfg.cruiseSpeed;

    // 能量
    var eHover = pHover * cfg.hoverTimeS / 3600.0;
    var eClimb = pClimb * tClimb / 3600.0;
    var eCruise = pCruise * tCruise / 3600.0;
    var eMission = (eHover + eClimb + eCruise) * (1.0 + cfg.reserveFraction);
    var eRequired = usableEnergy(eMission, cfg.batteryUsableFraction);
    var mBattery = eRequired / cfg.batterySpecificEnergy;

    // 重量
    var pMaxShaft = Math.max(
      pHover / (cfg.etaMotor * cfg.etaTransmission),
      pClimb / (cfg.etaMotor * cfg.etaTransmission),
      pCruise / cfg.etaMotor
    );
    var maxThrust = w * cfg.thrustToWeight;

    var mStructure = cfg.structureFraction * mtow;
    var mPropulsion = pMaxShaft / cfg.motorSpecificPower;
    var mRotor = maxThrust / cfg.rotorSpecificThrust;

    var total = mStructure + mPropulsion + mRotor + cfg.avionicsMass + cfg.payloadMass + mBattery;

    var cdInduced = polar.cl * polar.cl / (Math.PI * cfg.aspectRatio * cfg.oswaldE);

    return {
      totalMass: total,
      force: w,
      hoverPower: pHover,
      climbPower: pClimb,
      cruisePower: pCruise,
      maxShaftPower: pMaxShaft,
      installedPower: pMaxShaft * cfg.powerMargin,
      massBreakdown: {
        structure: mStructure,
        propulsion: mPropulsion,
        rotor: mRotor,
        avionics: cfg.avionicsMass,
        payload: cfg.payloadMass,
        battery: mBattery
      },
      wingLoading: w / cfg.wingArea,
      diskLoading: w / area,
      clCruise: polar.cl,
      liftToDrag: polar.liftToDrag,
      inducedRatio: polar.cd > 0 ? cdInduced / polar.cd : 0.0,
      energyBreakdown: { hover: eHover, climb: eClimb, cruise: eCruise },
      missionEnergy: eMission,
      requiredEnergy: eRequired,
      timeBreakdown: { hover: cfg.hoverTimeS, climb: tClimb, cruise: tCruise },
      diskArea: area
    };
  }

  // ------------------------------------------------------------ 求解主函数
  function size(cfg) {
    var mtow = cfg.mtowGuess;
    var residual = NaN;
    var last = null;

    for (var it = 1; it <= cfg.maxIter; it++) {
      last = evaluate(cfg, mtow);
      residual = last.totalMass - mtow;
      mtow += cfg.relax * residual;

      if (!isFinite(mtow) || mtow <= 0) {
        return makeResult(cfg, false, it, residual,
          '迭代发散：任务需求超出能量可行性边界', NaN, evaluate(cfg, cfg.mtowGuess), 0);
      }
      if (Math.abs(residual) / Math.max(mtow, 1e-9) < cfg.tol) {
        var data = evaluate(cfg, mtow);
        var warning = '';
        if (data.clCruise > cfg.clMax) {
          warning = '巡航升力系数 ' + data.clCruise.toFixed(2) +
            ' 超过可用上限 ' + cfg.clMax.toFixed(2) + '，需加大机翼面积或提高巡航速度';
        }
        return makeResult(cfg, true, it, residual, warning, mtow, data, mtow);
      }
    }
    return makeResult(cfg, false, cfg.maxIter, residual,
      '达到最大迭代次数 ' + cfg.maxIter + ' 仍未收敛', mtow, last, 0);
  }

  function makeResult(cfg, converged, iterations, residual, warning, mtow, data, span) {
    return {
      converged: converged,
      iterations: iterations,
      residual: residual,
      warning: warning,
      mtow: mtow,
      massBreakdown: data.massBreakdown,
      hoverPower: data.hoverPower,
      climbPower: data.climbPower,
      cruisePower: data.cruisePower,
      maxShaftPower: data.maxShaftPower,
      installedPower: data.installedPower,
      wingLoading: data.wingLoading,
      diskLoading: data.diskLoading,
      clCruise: data.clCruise,
      liftToDrag: data.liftToDrag,
      inducedRatio: data.inducedRatio,
      energyBreakdown: data.energyBreakdown,
      missionEnergy: data.missionEnergy,
      requiredEnergy: data.requiredEnergy,
      timeBreakdown: data.timeBreakdown
    };
  }

  // -------------------------------------------------------- 纯悬停航时
  function hoverEnduranceMinutes(cfg, allUpMass, batteryMass) {
    var w = allUpMass * G0;
    var rhoSL = density(0.0);
    var area = diskArea(cfg.nRotor, cfg.rotorRadius);
    var pHover = toElectricPower(
      hoverShaftPower(w, rhoSL, area, cfg.fm),
      cfg.etaMotor, cfg.etaTransmission
    );
    var avail = batteryMass * cfg.batterySpecificEnergy * cfg.batteryUsableFraction;
    return (avail * 3600.0 / pHover) / 60.0;
  }

  // ------------------------------------------------------------ 参数扫描
  function linspace(start, stop, n) {
    if (n < 2) return [start];
    var step = (stop - start) / (n - 1);
    var out = [];
    for (var i = 0; i < n; i++) out.push(start + i * step);
    return out;
  }

  function withParams(cfg, overrides) {
    var out = {};
    for (var k in cfg) if (Object.prototype.hasOwnProperty.call(cfg, k)) out[k] = cfg[k];
    for (var j in overrides) if (Object.prototype.hasOwnProperty.call(overrides, j)) out[j] = overrides[j];
    return out;
  }

  /** 单参数扫描：field 为配置字段名 */
  function sweep1d(cfg, field, values) {
    var out = [];
    for (var i = 0; i < values.length; i++) {
      var ov = {};
      ov[field] = values[i];
      var res = size(withParams(cfg, ov));
      out.push({
        value: values[i],
        mtow: res.mtow,
        batteryMass: res.massBreakdown.battery,
        diskLoading: res.diskLoading,
        wingLoading: res.wingLoading,
        installedPower: res.installedPower / 1000.0,
        liftToDrag: res.liftToDrag,
        converged: res.converged
      });
    }
    return out;
  }

  /** 敏感性分析：对可调参数施加 ±delta 扰动，按 MTOW 相对变化排序 */
  // 与 Python 端 sweep._tunable_fields() 保持一致，顺序即 dataclass 字段顺序
  var TUNABLE = [
    'nRotor', 'rotorRadius', 'fm', 'etaMotor', 'etaTransmission',
    'wingArea', 'aspectRatio', 'cd0', 'oswaldE',
    'structureFraction', 'avionicsMass',
    'motorSpecificPower', 'rotorSpecificThrust', 'powerMargin', 'thrustToWeight',
    'batterySpecificEnergy', 'batteryUsableFraction',
    'climbRate', 'climbAltitude',
    'cruiseSpeed', 'cruiseAltitude', 'cruisePropEfficiency'
  ];

  var FIELD_LABELS = {
    nRotor: '旋翼数量',
    rotorRadius: '桨半径',
    fm: '品质因数 FM',
    etaMotor: '电机效率',
    etaTransmission: '传动效率',
    wingArea: '机翼面积',
    aspectRatio: '展弦比',
    cd0: '零升阻力系数',
    oswaldE: '奥斯瓦尔德因子',
    structureFraction: '结构重量系数',
    avionicsMass: '航电重量',
    motorSpecificPower: '电机功率密度',
    rotorSpecificThrust: '旋翼推力密度',
    powerMargin: '装机功率裕度',
    thrustToWeight: '设计推重比',
    batterySpecificEnergy: '电池比能量',
    batteryUsableFraction: '可用放电深度',
    climbRate: '爬升率',
    climbAltitude: '爬升高度',
    cruiseSpeed: '巡航速度',
    cruiseAltitude: '巡航高度',
    cruisePropEfficiency: '巡航螺旋桨效率'
  };

  function sensitivity(cfg, delta) {
    delta = delta === undefined ? 0.10 : delta;
    var base = size(cfg).mtow;
    var rows = [];
    for (var i = 0; i < TUNABLE.length; i++) {
      var name = TUNABLE[i];
      var cur = cfg[name];
      if (typeof cur !== 'number' || cur === 0) continue;
      var hiOv = {}, loOv = {};
      hiOv[name] = cur * (1 + delta);
      loOv[name] = cur * (1 - delta);
      var vHi = size(withParams(cfg, hiOv)).mtow;
      var vLo = size(withParams(cfg, loOv)).mtow;
      var swing, dir;
      if (!isFinite(vHi) || !isFinite(vLo) || base === 0) {
        swing = Infinity; dir = '?';
      } else {
        swing = 100 * Math.max(Math.abs(vHi - base), Math.abs(vLo - base)) / Math.abs(base);
        if (vHi > base && vLo < base) dir = '+1';
        else if (vHi < base && vLo > base) dir = '-1';
        else dir = '?';
      }
      rows.push({
        field: name,
        label: FIELD_LABELS[name] || name,
        base: base,
        plus: vHi,
        minus: vLo,
        swingPct: swing,
        direction: dir
      });
    }
    rows.sort(function (a, b) {
      var av = isFinite(a.swingPct) ? a.swingPct : 1e18;
      var bv = isFinite(b.swingPct) ? b.swingPct : 1e18;
      return bv - av;
    });
    return rows;
  }

  return {
    G0: G0,
    defaultConfig: defaultConfig,
    density: density,
    pressure: pressure,
    temperature: temperature,
    diskArea: diskArea,
    liftCurveSlope: liftCurveSlope,
    wingPolar: wingPolar,
    evaluate: evaluate,
    size: size,
    hoverEnduranceMinutes: hoverEnduranceMinutes,
    sweep1d: sweep1d,
    sensitivity: sensitivity,
    linspace: linspace,
    withParams: withParams,
    TUNABLE: TUNABLE,
    FIELD_LABELS: FIELD_LABELS
  };
});
