"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.runPythonScript = void 0;
const child_process_1 = require("child_process");
const log_1 = require("./log");
const logger = log_1.getLogger('pythonScript');
const python = process.platform === 'win32' ? 'python.exe' : 'python3';
async function runPythonScript(script, logTag) {
    const proc = child_process_1.spawn(python, ['-c', script]);
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (data) => { stdout += data; });
    proc.stderr.on('data', (data) => { stderr += data; });
    const procPromise = new Promise((resolve, reject) => {
        proc.on('error', (err) => { reject(err); });
        proc.on('exit', () => { resolve(); });
    });
    await procPromise;
    if (stderr) {
        if (logTag) {
            logger.warning(`Python script [${logTag}] has stderr:`, stderr);
        }
        else {
            logger.warning('Python script has stderr.');
            logger.warning('  script:', script);
            logger.warning('  stderr:', stderr);
        }
    }
    return stdout;
}
exports.runPythonScript = runPythonScript;
