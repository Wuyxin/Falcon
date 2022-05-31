"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.runGpuMetricsCollector = exports.getGpuMetricsCollectorBashScriptContent = exports.getScriptName = exports.tarAdd = exports.setEnvironmentVariable = exports.execKill = exports.execRemove = exports.execTail = exports.runScript = exports.execNewFile = exports.execCopydir = exports.execMkdir = exports.validateCodeDir = exports.listDirWithIgnoredFiles = void 0;
const child_process_promise_1 = __importDefault(require("child-process-promise"));
const child_process_1 = __importDefault(require("child_process"));
const fs_1 = __importDefault(require("fs"));
const ignore_1 = __importDefault(require("ignore"));
const path_1 = __importDefault(require("path"));
const tar_1 = __importDefault(require("tar"));
const log_1 = require("common/log");
const typescript_string_operations_1 = require("typescript-string-operations");
const gpuData_1 = require("./gpuData");
function* listDirWithIgnoredFiles(root, relDir, ignoreFiles) {
    let ignoreFile = undefined;
    const source = path_1.default.join(root, relDir);
    if (fs_1.default.existsSync(path_1.default.join(source, '.nniignore'))) {
        ignoreFile = path_1.default.join(source, '.nniignore');
        ignoreFiles.push(ignoreFile);
    }
    const ig = ignore_1.default();
    ignoreFiles.forEach((i) => ig.add(fs_1.default.readFileSync(i).toString()));
    for (const d of fs_1.default.readdirSync(source)) {
        const entry = path_1.default.join(relDir, d);
        if (ig.ignores(entry))
            continue;
        const entryStat = fs_1.default.statSync(path_1.default.join(root, entry));
        if (entryStat.isDirectory()) {
            yield entry;
            yield* listDirWithIgnoredFiles(root, entry, ignoreFiles);
        }
        else if (entryStat.isFile())
            yield entry;
    }
    if (ignoreFile !== undefined) {
        ignoreFiles.pop();
    }
}
exports.listDirWithIgnoredFiles = listDirWithIgnoredFiles;
async function validateCodeDir(codeDir) {
    let fileCount = 0;
    let fileTotalSize = 0;
    for (const relPath of listDirWithIgnoredFiles(codeDir, '', [])) {
        const d = path_1.default.join(codeDir, relPath);
        fileCount += 1;
        fileTotalSize += fs_1.default.statSync(d).size;
        if (fileCount > 2000) {
            throw new Error(`Too many files and directories (${fileCount} already scanned) in ${codeDir},`
                + ` please check if it's a valid code dir`);
        }
        if (fileTotalSize > 300 * 1024 * 1024) {
            throw new Error(`File total size too large in code dir (${fileTotalSize} bytes already scanned, exceeds 300MB).`);
        }
        const fileNameValid = relPath.split(path_1.default.sep).every(fpart => (fpart.match('^[a-z0-9A-Z._-]*$') !== null));
        if (!fileNameValid) {
            const message = [
                `File ${relPath} in directory ${codeDir} contains spaces or special characters in its name.`,
                'This might cause problem when uploading to cloud or remote machine.',
                'If you encounter any error, please report an issue: https://github.com/microsoft/nni/issues'
            ].join(' ');
            log_1.getLogger('validateCodeDir').warning(message);
        }
    }
    return fileCount;
}
exports.validateCodeDir = validateCodeDir;
async function execMkdir(directory, share = false) {
    if (process.platform === 'win32') {
        await child_process_promise_1.default.exec(`powershell.exe New-Item -Path "${directory}" -ItemType "directory" -Force`);
    }
    else if (share) {
        await child_process_promise_1.default.exec(`(umask 0; mkdir -p '${directory}')`);
    }
    else {
        await child_process_promise_1.default.exec(`mkdir -p '${directory}'`);
    }
    return Promise.resolve();
}
exports.execMkdir = execMkdir;
async function execCopydir(source, destination) {
    if (!fs_1.default.existsSync(destination))
        await fs_1.default.promises.mkdir(destination);
    for (const relPath of listDirWithIgnoredFiles(source, '', [])) {
        const sourcePath = path_1.default.join(source, relPath);
        const destPath = path_1.default.join(destination, relPath);
        if (fs_1.default.statSync(sourcePath).isDirectory()) {
            if (!fs_1.default.existsSync(destPath)) {
                await fs_1.default.promises.mkdir(destPath);
            }
        }
        else {
            log_1.getLogger('execCopydir').debug(`Copying file from ${sourcePath} to ${destPath}`);
            await fs_1.default.promises.copyFile(sourcePath, destPath);
        }
    }
    return Promise.resolve();
}
exports.execCopydir = execCopydir;
async function execNewFile(filename) {
    if (process.platform === 'win32') {
        await child_process_promise_1.default.exec(`powershell.exe New-Item -Path "${filename}" -ItemType "file" -Force`);
    }
    else {
        await child_process_promise_1.default.exec(`touch '${filename}'`);
    }
    return Promise.resolve();
}
exports.execNewFile = execNewFile;
function runScript(filePath) {
    if (process.platform === 'win32') {
        return child_process_1.default.exec(`powershell.exe -ExecutionPolicy Bypass -file "${filePath}"`);
    }
    else {
        return child_process_1.default.exec(`bash '${filePath}'`);
    }
}
exports.runScript = runScript;
async function execTail(filePath) {
    let cmdresult;
    if (process.platform === 'win32') {
        cmdresult = await child_process_promise_1.default.exec(`powershell.exe Get-Content "${filePath}" -Tail 1`);
    }
    else {
        cmdresult = await child_process_promise_1.default.exec(`tail -n 1 '${filePath}'`);
    }
    return Promise.resolve(cmdresult);
}
exports.execTail = execTail;
async function execRemove(directory) {
    if (process.platform === 'win32') {
        await child_process_promise_1.default.exec(`powershell.exe Remove-Item "${directory}" -Recurse -Force`);
    }
    else {
        await child_process_promise_1.default.exec(`rm -rf '${directory}'`);
    }
    return Promise.resolve();
}
exports.execRemove = execRemove;
async function execKill(pid) {
    if (process.platform === 'win32') {
        await child_process_promise_1.default.exec(`cmd.exe /c taskkill /PID ${pid} /T /F`);
    }
    else {
        await child_process_promise_1.default.exec(`pkill -P ${pid}`);
    }
    return Promise.resolve();
}
exports.execKill = execKill;
function setEnvironmentVariable(variable) {
    if (process.platform === 'win32') {
        return `$env:${variable.key}="${variable.value}"`;
    }
    else {
        return `export ${variable.key}='${variable.value}'`;
    }
}
exports.setEnvironmentVariable = setEnvironmentVariable;
async function tarAdd(tarPath, sourcePath) {
    const fileList = [];
    for (const d of listDirWithIgnoredFiles(sourcePath, '', [])) {
        fileList.push(d);
    }
    tar_1.default.create({
        gzip: true,
        file: tarPath,
        sync: true,
        cwd: sourcePath,
    }, fileList);
    return Promise.resolve();
}
exports.tarAdd = tarAdd;
function getScriptName(fileNamePrefix) {
    if (process.platform === 'win32') {
        return typescript_string_operations_1.String.Format('{0}.ps1', fileNamePrefix);
    }
    else {
        return typescript_string_operations_1.String.Format('{0}.sh', fileNamePrefix);
    }
}
exports.getScriptName = getScriptName;
function getGpuMetricsCollectorBashScriptContent(scriptFolder) {
    return `echo $$ > ${scriptFolder}/pid ; METRIC_OUTPUT_DIR=${scriptFolder} python3 -m nni.tools.gpu_tool.gpu_metrics_collector \
1>${scriptFolder}/stdout 2>${scriptFolder}/stderr`;
}
exports.getGpuMetricsCollectorBashScriptContent = getGpuMetricsCollectorBashScriptContent;
function runGpuMetricsCollector(scriptFolder) {
    if (process.platform === 'win32') {
        const scriptPath = path_1.default.join(scriptFolder, 'gpu_metrics_collector.ps1');
        const content = typescript_string_operations_1.String.Format(gpuData_1.GPU_INFO_COLLECTOR_FORMAT_WINDOWS, scriptFolder, path_1.default.join(scriptFolder, 'pid'));
        fs_1.default.writeFile(scriptPath, content, { encoding: 'utf8' }, () => { runScript(scriptPath); });
    }
    else {
        child_process_1.default.exec(getGpuMetricsCollectorBashScriptContent(scriptFolder), { shell: '/bin/bash' });
    }
}
exports.runGpuMetricsCollector = runGpuMetricsCollector;
