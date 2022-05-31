"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.toCudaVisibleDevices = exports.toMegaBytes = exports.toSeconds = void 0;
const timeUnits = { d: 24 * 3600, h: 3600, m: 60, s: 1 };
const sizeUnits = { tb: 1024 ** 4, gb: 1024 ** 3, mb: 1024 ** 2, kb: 1024, b: 1 };
function toUnit(value, targetUnit, allUnits) {
    if (typeof value === 'number') {
        return value;
    }
    value = value.toLowerCase();
    for (const [unit, factor] of Object.entries(allUnits)) {
        if (value.endsWith(unit)) {
            const digits = value.slice(0, -unit.length);
            const num = Number(digits) * factor;
            return Math.ceil(num / allUnits[targetUnit]);
        }
    }
    throw new Error(`Bad unit in "${value}"`);
}
function toSeconds(time) {
    return toUnit(time, 's', timeUnits);
}
exports.toSeconds = toSeconds;
function toMegaBytes(size) {
    return toUnit(size, 'mb', sizeUnits);
}
exports.toMegaBytes = toMegaBytes;
function toCudaVisibleDevices(gpuIndices) {
    return gpuIndices === undefined ? '' : gpuIndices.join(',');
}
exports.toCudaVisibleDevices = toCudaVisibleDevices;
