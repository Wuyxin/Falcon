'use strict';
Object.defineProperty(exports, "__esModule", { value: true });
var ContainerConfig = (function () {
    function ContainerConfig() {
    }
    ContainerConfig.addSource = function (patterns, baseDir) {
        var requireGlob = require('require-glob');
        baseDir = baseDir || process.cwd();
        requireGlob.sync(patterns, {
            cwd: baseDir
        });
    };
    return ContainerConfig;
}());
exports.ContainerConfig = ContainerConfig;
//# sourceMappingURL=container-config.js.map