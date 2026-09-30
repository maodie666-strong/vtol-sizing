/**
 * 浏览器端代理模型推理的一致性验证。
 *
 * 用法：
 *   1) 先由 Python 导出测试点：python surrogate/dump_predictions.py
 *   2) 运行本脚本：           node vtol-web/verify_surrogate.js
 *
 * 校验内容：同一套权重，Python(numpy) 与 JavaScript 的推理结果是否一致。
 * 由于导出权重时保留了 7 位有效数字，两者应当一致到 1e-6 量级。
 */

const fs = require('fs');
const path = require('path');
const { Surrogate } = require('./surrogate.js');

const WEB_DIR = __dirname;
const weightsPath = path.join(WEB_DIR, 'surrogate.json');
const pointsPath = path.join(WEB_DIR, 'surrogate_test_points.json');

for (const p of [weightsPath, pointsPath]) {
  if (!fs.existsSync(p)) {
    console.error('缺少文件：' + p);
    console.error('请先运行：python surrogate/train.py && python surrogate/dump_predictions.py');
    process.exit(1);
  }
}

const payload = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
const points = JSON.parse(fs.readFileSync(pointsPath, 'utf8'));

const net = new Surrogate(payload);
const outputNames = points.output_names;

let pass = 0;
let fail = 0;
let maxErr = 0;
const perOutput = {};
outputNames.forEach((n) => { perOutput[n] = { max: 0, sum: 0, n: 0 }; });

const TOL = 1e-5;   // 相对容差（权重已截断到 7 位有效数字）

points.cases.forEach((c, idx) => {
  const got = net.predict(c.inputs);
  outputNames.forEach((name) => {
    const expected = c.expected[name];
    const actual = got[name];
    const rel = expected === 0 ? Math.abs(actual) : Math.abs(actual - expected) / Math.abs(expected);
    perOutput[name].max = Math.max(perOutput[name].max, rel);
    perOutput[name].sum += rel;
    perOutput[name].n += 1;
    maxErr = Math.max(maxErr, rel);
    if (rel <= TOL) pass += 1; else {
      fail += 1;
      if (fail <= 5) {
        console.log(`  [FAIL] case ${idx} ${name}: PY=${expected.toExponential(8)} `
          + `JS=${actual.toExponential(8)} err=${rel.toExponential(2)}`);
      }
    }
  });
});

console.log('');
console.log('=== 逐输出量最大相对误差 ===');
console.log('  ' + 'output'.padEnd(20) + 'max rel err'.padStart(14) + 'mean rel err'.padStart(16));
Object.keys(perOutput).forEach((name) => {
  const s = perOutput[name];
  console.log('  ' + name.padEnd(20)
    + s.max.toExponential(2).padStart(14)
    + (s.sum / s.n).toExponential(2).padStart(16));
});

console.log('');
console.log('=========================================================');
console.log(`  测试点 ${points.cases.length} 个，输出量 ${outputNames.length} 个`);
console.log(`  比对项 ${pass + fail} 项：通过 ${pass}，失败 ${fail}`);
console.log(`  最大相对误差 ${maxErr.toExponential(2)}（容差 ${TOL.toExponential(0)}）`);
console.log(fail === 0
  ? '  结论：浏览器端推理与 Python 训练端数值一致'
  : '  结论：存在不一致，需排查');
console.log('=========================================================');

process.exit(fail === 0 ? 0 : 1);
