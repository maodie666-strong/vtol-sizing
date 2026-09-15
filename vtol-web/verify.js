/**
 * JS 移植版与 Python 内核的跨语言一致性验证。
 *
 * 用法：
 *   1) 先在 vtol-sizing 目录生成基准：python vtol-web/dump_reference.py
 *   2) 运行本脚本：node vtol-web/verify.js
 *
 * 两者共用同一套物理模型与参数，结果应在浮点误差内一致（容差 1e-9）。
 */

const fs = require('fs');
const path = require('path');
const VTOL = require('./model.js');

const REF_PATH = path.join(__dirname, 'reference.json');
if (!fs.existsSync(REF_PATH)) {
  console.error('找不到 reference.json，请先运行：python vtol-web/dump_reference.py');
  process.exit(1);
}
const ref = JSON.parse(fs.readFileSync(REF_PATH, 'utf8'));

const TOL = 1e-9;
let pass = 0;
let fail = 0;

function check(label, actual, expected, unit) {
  const relErr = expected === 0
    ? Math.abs(actual)
    : Math.abs(actual - expected) / Math.abs(expected);
  const ok = relErr <= TOL;
  if (ok) pass++; else fail++;
  console.log(
    '  [' + (ok ? 'PASS' : 'FAIL') + '] ' + label.padEnd(24) +
    'PY=' + expected.toExponential(8).padStart(16) +
    '  JS=' + actual.toExponential(8).padStart(16) +
    '  err=' + relErr.toExponential(1) + (unit ? '  ' + unit : '')
  );
}

function rule(t) {
  console.log('');
  console.log('=== ' + t + ' ===');
}

// ------------------------------------------------------------------ 案例 1
rule('案例 1  中型载货 eVTOL（对标 demo.py）');
const cfg = VTOL.defaultConfig();
const r = VTOL.size(cfg);
const c1 = ref.case1;
const mb = c1.mass_breakdown;

check('收敛标志', r.converged ? 1 : 0, c1.converged ? 1 : 0);
check('迭代次数', r.iterations, c1.iterations, '次');
check('MTOW', r.mtow, c1.mtow, 'kg');
check('结构重量', r.massBreakdown.structure, mb.structure, 'kg');
check('动力系统重量', r.massBreakdown.propulsion, mb.propulsion, 'kg');
check('旋翼系统重量', r.massBreakdown.rotor, mb.rotor, 'kg');
check('航电重量', r.massBreakdown.avionics, mb.avionics, 'kg');
check('有效载荷', r.massBreakdown.payload, mb.payload, 'kg');
check('电池重量', r.massBreakdown.battery, mb.battery, 'kg');
check('悬停电功率', r.hoverPower, c1.hover_power_w, 'W');
check('爬升电功率', r.climbPower, c1.climb_power_w, 'W');
check('巡航电功率', r.cruisePower, c1.cruise_power_w, 'W');
check('装机功率', r.installedPower, c1.installed_power_w, 'W');
check('翼载', r.wingLoading, c1.wing_loading, 'N/m2');
check('桨盘载荷', r.diskLoading, c1.disk_loading, 'N/m2');
check('巡航升力系数', r.clCruise, c1.cl_cruise);
check('巡航升阻比', r.liftToDrag, c1.lift_to_drag);
check('诱导阻力占比', r.inducedRatio, c1.induced_ratio);
check('悬停段能量', r.energyBreakdown.hover, c1.energy_breakdown_wh.hover, 'Wh');
check('爬升段能量', r.energyBreakdown.climb, c1.energy_breakdown_wh.climb, 'Wh');
check('巡航段能量', r.energyBreakdown.cruise, c1.energy_breakdown_wh.cruise, 'Wh');
check('含储备能量', r.missionEnergy, c1.mission_energy_wh, 'Wh');
check('需装机能量', r.requiredEnergy, c1.required_energy_wh, 'Wh');

// ------------------------------------------------------------------ 案例 2
rule('案例 2  小四旋翼悬停航时');
const quad = VTOL.withParams(VTOL.defaultConfig(), {
  nRotor: 4,
  rotorRadius: 0.1397,
  fm: 0.62,
  etaMotor: 0.85,
  etaTransmission: 1.00,
  batterySpecificEnergy: 220.0
});
check('悬停航时', VTOL.hoverEnduranceMinutes(quad, 2.0, 0.5), ref.case2.endurance_min, 'min');

// -------------------------------------------------------------- 敏感性分析
rule('案例 3  参数敏感性排序');

/** camelCase -> snake_case，用于跨语言比较字段名 */
function toSnake(s) {
  return s.replace(/[A-Z]/g, (m) => '_' + m.toLowerCase());
}

const sens = VTOL.sensitivity(cfg, 0.10);
const actualOrder = sens.map((s) => toSnake(s.field));
const orderOk = JSON.stringify(actualOrder) === JSON.stringify(ref.sensitivity_order);
if (orderOk) pass++; else fail++;
console.log('  [' + (orderOk ? 'PASS' : 'FAIL') + '] 排序与 Python 完全一致（' + actualOrder.length + ' 项）');
if (!orderOk) {
  console.log('    Python: ' + ref.sensitivity_order.join(' > '));
  console.log('    JS    : ' + actualOrder.join(' > '));
}
sens.forEach((s) => {
  const exp = ref.sensitivity_swing[toSnake(s.field)];
  if (exp === undefined) return;
  check('  ' + s.label, s.swingPct, exp, '%');
});

// ------------------------------------------------------------------ 扫描
rule('案例 4  桨半径扫描');
const radii = VTOL.linspace(0.60, 1.30, 15);
const scan = VTOL.sweep1d(cfg, 'rotorRadius', radii);
ref.scan_rotor_radius.forEach((item, i) => {
  check('  r=' + item.radius.toFixed(2) + ' m', scan[i].mtow, item.mtow, 'kg');
});
const monotone = scan.every((s, i) => i === 0 || s.mtow <= scan[i - 1].mtow + 1e-9);
if (monotone) pass++; else fail++;
console.log('  [' + (monotone ? 'PASS' : 'FAIL') + '] MTOW 随桨半径单调下降');

// ------------------------------------------------------------------ 汇总
console.log('');
console.log('=========================================================');
console.log('  通过 ' + pass + ' 项，失败 ' + fail + ' 项');
console.log('  容差 ' + TOL.toExponential(0) + '（相对误差）');
console.log(fail === 0
  ? '  结论：JS 移植版与 Python 内核数值完全一致'
  : '  结论：存在不一致项，需排查');
console.log('=========================================================');
process.exit(fail === 0 ? 0 : 1);
