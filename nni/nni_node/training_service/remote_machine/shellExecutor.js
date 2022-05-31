"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ShellExecutor = void 0;
const assert_1 = __importDefault(require("assert"));
const fs_1 = __importDefault(require("fs"));
const os_1 = __importDefault(require("os"));
const path_1 = __importDefault(require("path"));
const ssh2_1 = require("ssh2");
const ts_deferred_1 = require("ts-deferred");
const log_1 = require("common/log");
const utils_1 = require("common/utils");
const util_1 = require("../common/util");
const linuxCommands_1 = require("./extends/linuxCommands");
const windowsCommands_1 = require("./extends/windowsCommands");
const errors_1 = require("common/errors");
class ShellExecutor {
    name = "";
    lineBreaker = new RegExp(`[\r\n]+`);
    maxUsageCount = 5;
    osCommands;
    usedCount = 0;
    sshClient;
    log;
    tempPath = "";
    channelDefaultOutputs = [];
    pythonPath;
    isWindows = false;
    constructor() {
        this.log = log_1.getLogger('ShellExecutor');
        this.sshClient = new ssh2_1.Client();
    }
    async initialize(rmMeta) {
        const deferred = new ts_deferred_1.Deferred();
        const connectConfig = {
            host: rmMeta.config.host,
            port: rmMeta.config.port,
            username: rmMeta.config.user,
            tryKeyboard: true,
        };
        this.pythonPath = rmMeta.config.pythonPath;
        this.name = `${rmMeta.config.user}@${rmMeta.config.host}:${rmMeta.config.port}`;
        if (rmMeta.config.password !== undefined) {
            connectConfig.password = rmMeta.config.password;
        }
        else if (rmMeta.config.sshKeyFile !== undefined) {
            if (!fs_1.default.existsSync(rmMeta.config.sshKeyFile)) {
                deferred.reject(new Error(`${rmMeta.config.sshKeyFile} does not exist.`));
            }
            const privateKey = fs_1.default.readFileSync(rmMeta.config.sshKeyFile, 'utf8');
            connectConfig.privateKey = privateKey;
            connectConfig.passphrase = rmMeta.config.sshPassphrase;
        }
        else {
            deferred.reject(new Error(`No valid passwd or sshKeyPath is configed.`));
        }
        this.sshClient.on('ready', async () => {
            const result = await this.execute("ver");
            if (result.exitCode == 0 && result.stdout.search("Windows") > -1) {
                this.osCommands = new windowsCommands_1.WindowsCommands();
                this.isWindows = true;
                let defaultResult = await this.execute("");
                if (defaultResult.stdout !== "") {
                    deferred.reject(new Error(`The windows remote node shouldn't output welcome message, below content should be removed from the command window! \n` +
                        `${defaultResult.stdout}`));
                }
                defaultResult = await this.execute("powershell -command \"\"");
                if (defaultResult.stdout !== "") {
                    this.channelDefaultOutputs.push(defaultResult.stdout);
                }
                this.log.debug(`set channelDefaultOutput to "${this.channelDefaultOutputs}"`);
                const commandResult = await this.execute("echo %TEMP%");
                this.tempPath = commandResult.stdout.replace(this.lineBreaker, "");
            }
            else {
                this.osCommands = new linuxCommands_1.LinuxCommands();
                this.tempPath = "/tmp";
            }
            deferred.resolve();
        }).on('error', (err) => {
            deferred.reject(new Error(err.message));
        }).on("keyboard-interactive", (_name, _instructions, _lang, _prompts, finish) => {
            finish([rmMeta.config.password || '']);
        }).connect(connectConfig);
        return deferred.promise;
    }
    close() {
        this.sshClient.end();
    }
    addUsage() {
        let isAddedSuccess = false;
        if (this.usedCount < this.maxUsageCount) {
            this.usedCount++;
            isAddedSuccess = true;
        }
        return isAddedSuccess;
    }
    releaseUsage() {
        let canBeReleased = false;
        if (this.usedCount > 0) {
            this.usedCount--;
        }
        if (this.usedCount == 0) {
            canBeReleased = true;
        }
        return canBeReleased;
    }
    getScriptName(mainName) {
        if (this.osCommands === undefined) {
            throw new Error("osCommands must be initialized!");
        }
        return `${mainName}.${this.osCommands.getScriptExt()}`;
    }
    generateStartScript(workingDirectory, trialJobId, experimentId, trialSequenceId, isMultiPhase, command, nniManagerAddress, nniManagerPort, nniManagerVersion, logCollection, cudaVisibleSetting) {
        if (this.osCommands === undefined) {
            throw new Error("osCommands must be initialized!");
        }
        const jobIdFileName = this.joinPath(workingDirectory, '.nni', 'jobpid');
        const exitCodeFile = this.joinPath(workingDirectory, '.nni', 'code');
        const codeDir = this.getRemoteCodePath(experimentId);
        return this.osCommands.generateStartScript(workingDirectory, trialJobId, experimentId, trialSequenceId, isMultiPhase, jobIdFileName, command, nniManagerAddress, nniManagerPort, nniManagerVersion, logCollection, exitCodeFile, codeDir, cudaVisibleSetting);
    }
    generateGpuStatsScript(experimentId) {
        if (this.osCommands === undefined) {
            throw new Error("osCommands must be initialized!");
        }
        return this.osCommands.generateGpuStatsScript(this.getRemoteScriptsPath(experimentId));
    }
    getTempPath() {
        if (this.tempPath === "") {
            throw new Error("tempPath must be initialized!");
        }
        return this.tempPath;
    }
    async getCurrentPath() {
        const commandText = this.osCommands && this.osCommands.getCurrentPath();
        const commandResult = await this.execute(commandText);
        if (commandResult.exitCode == 0) {
            return commandResult.stdout;
        }
        else {
            throw Error(commandResult.stderr);
        }
    }
    getRemoteScriptsPath(experimentId) {
        return this.joinPath(this.getRemoteExperimentRootDir(experimentId), 'scripts');
    }
    getRemoteCodePath(experimentId) {
        return this.joinPath(this.getRemoteExperimentRootDir(experimentId), 'nni-code');
    }
    getRemoteExperimentRootDir(experimentId) {
        return this.joinPath(this.tempPath, 'nni-experiments', experimentId);
    }
    joinPath(...paths) {
        if (!this.osCommands) {
            throw new Error("osCommands must be initialized!");
        }
        return this.osCommands.joinPath(...paths);
    }
    async createFolder(folderName, sharedFolder = false) {
        const commandText = this.osCommands && this.osCommands.createFolder(folderName, sharedFolder);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode == 0;
        return result;
    }
    async allowPermission(isRecursive = false, ...folders) {
        const commandText = this.osCommands && this.osCommands.allowPermission(isRecursive, ...folders);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode == 0;
        return result;
    }
    async removeFolder(folderName, isRecursive = false, isForce = true) {
        const commandText = this.osCommands && this.osCommands.removeFolder(folderName, isRecursive, isForce);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode == 0;
        return result;
    }
    async removeFiles(folderOrFileName, filePattern = "") {
        const commandText = this.osCommands && this.osCommands.removeFiles(folderOrFileName, filePattern);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode == 0;
        return result;
    }
    async readLastLines(fileName, lineCount = 1) {
        const commandText = this.osCommands && this.osCommands.readLastLines(fileName, lineCount);
        const commandResult = await this.execute(commandText);
        let result = "";
        if (commandResult !== undefined && commandResult.stdout !== undefined && commandResult.stdout.length > 0) {
            result = commandResult.stdout;
        }
        return result;
    }
    async isProcessAlive(pidFileName) {
        const commandText = this.osCommands && this.osCommands.isProcessAliveCommand(pidFileName);
        const commandResult = await this.execute(commandText);
        const result = this.osCommands && this.osCommands.isProcessAliveProcessOutput(commandResult);
        return result !== undefined ? result : false;
    }
    async killChildProcesses(pidFileName, killSelf = false) {
        const commandText = this.osCommands && this.osCommands.killChildProcesses(pidFileName, killSelf);
        const commandResult = await this.execute(commandText);
        return commandResult.exitCode == 0;
    }
    async fileExist(filePath) {
        const commandText = this.osCommands && this.osCommands.fileExistCommand(filePath);
        const commandResult = await this.execute(commandText);
        return commandResult.stdout !== undefined && commandResult.stdout.trim() === 'True';
    }
    async extractFile(tarFileName, targetFolder) {
        const commandText = this.osCommands && this.osCommands.extractFile(tarFileName, targetFolder);
        const commandResult = await this.execute(commandText);
        return commandResult.exitCode == 0;
    }
    async executeScript(script, isFile = false, isInteractive = false) {
        const commandText = this.osCommands && this.osCommands.executeScript(script, isFile);
        const commandResult = await this.execute(commandText, undefined, isInteractive);
        return commandResult;
    }
    async copyFileToRemote(localFilePath, remoteFilePath) {
        const commandIndex = utils_1.randomInt(10000);
        this.log.debug(`copyFileToRemote(${commandIndex}): localFilePath: ${localFilePath}, remoteFilePath: ${remoteFilePath}`);
        const deferred = new ts_deferred_1.Deferred();
        this.sshClient.sftp((err, sftp) => {
            if (err !== undefined && err !== null) {
                this.log.error(`copyFileToRemote(${commandIndex}): ${err}`);
                deferred.reject(err);
                return;
            }
            assert_1.default(sftp !== undefined);
            sftp.fastPut(localFilePath, remoteFilePath, (fastPutErr) => {
                sftp.end();
                if (fastPutErr !== undefined && fastPutErr !== null) {
                    this.log.error(`copyFileToRemote(${commandIndex}) fastPutErr: ${fastPutErr}, ${localFilePath}, ${remoteFilePath}`);
                    deferred.reject(fastPutErr);
                }
                else {
                    deferred.resolve(true);
                }
            });
        });
        return deferred.promise;
    }
    async copyDirectoryToRemote(localDirectory, remoteDirectory) {
        const tmpSuffix = utils_1.uniqueString(5);
        const localTarPath = path_1.default.join(os_1.default.tmpdir(), `nni_tmp_local_${tmpSuffix}.tar.gz`);
        if (!this.osCommands) {
            throw new Error("osCommands must be initialized!");
        }
        const remoteTarPath = this.osCommands.joinPath(this.tempPath, `nni_tmp_remote_${tmpSuffix}.tar.gz`);
        await this.createFolder(remoteDirectory);
        await util_1.tarAdd(localTarPath, localDirectory);
        await this.copyFileToRemote(localTarPath, remoteTarPath);
        await util_1.execRemove(localTarPath);
        await this.extractFile(remoteTarPath, remoteDirectory);
        await this.removeFiles(remoteTarPath);
    }
    async getRemoteFileContent(filePath) {
        const commandIndex = utils_1.randomInt(10000);
        this.log.debug(`getRemoteFileContent(${commandIndex}): filePath: ${filePath}`);
        const deferred = new ts_deferred_1.Deferred();
        this.sshClient.sftp((err, sftp) => {
            if (err !== undefined && err !== null) {
                this.log.error(`getRemoteFileContent(${commandIndex}) sftp: ${err}`);
                deferred.reject(new Error(`SFTP error: ${err}`));
                return;
            }
            try {
                const sftpStream = sftp.createReadStream(filePath);
                let dataBuffer = '';
                sftpStream.on('data', (data) => {
                    dataBuffer += data;
                })
                    .on('error', (streamErr) => {
                    sftp.end();
                    deferred.reject(new errors_1.NNIError(errors_1.NNIErrorNames.NOT_FOUND, streamErr.message));
                })
                    .on('end', () => {
                    sftp.end();
                    deferred.resolve(dataBuffer);
                });
            }
            catch (error) {
                this.log.error(`getRemoteFileContent(${commandIndex}): ${error.message}`);
                sftp.end();
                deferred.reject(new Error(`SFTP error: ${error.message}`));
            }
        });
        return deferred.promise;
    }
    async execute(command, processOutput = undefined, useShell = false) {
        const deferred = new ts_deferred_1.Deferred();
        let stdout = '';
        let stderr = '';
        let exitCode;
        const commandIndex = utils_1.randomInt(10000);
        if (this.osCommands !== undefined) {
            command = this.osCommands.setPythonPath(this.pythonPath, command);
        }
        this.log.debug(`remoteExeCommand(${commandIndex}): [${command}]`);
        useShell = useShell && !this.isWindows;
        const callback = (err, channel) => {
            if (err !== undefined && err !== null) {
                this.log.error(`remoteExeCommand(${commandIndex}): ${err.message}`);
                deferred.reject(err);
                return;
            }
            channel.on('data', (data) => {
                stdout += data;
            });
            channel.on('exit', (code) => {
                exitCode = code;
                if (this.channelDefaultOutputs.length > 0) {
                    let modifiedStdout = stdout;
                    this.channelDefaultOutputs.forEach(defaultOutput => {
                        if (modifiedStdout.startsWith(defaultOutput)) {
                            if (modifiedStdout.length > defaultOutput.length) {
                                modifiedStdout = modifiedStdout.substr(defaultOutput.length);
                            }
                            else if (modifiedStdout.length === defaultOutput.length) {
                                modifiedStdout = "";
                            }
                        }
                    });
                    stdout = modifiedStdout;
                }
                this.log.debug(`remoteExeCommand(${commandIndex}) exit(${exitCode})\nstdout: ${stdout}\nstderr: ${stderr}`);
                let result = {
                    stdout: stdout,
                    stderr: stderr,
                    exitCode: exitCode
                };
                if (processOutput != undefined) {
                    result = processOutput(result);
                }
                deferred.resolve(result);
            });
            channel.stderr.on('data', function (data) {
                stderr += data;
            });
            if (useShell) {
                channel.stdin.write(`${command}\n`);
                channel.end("exit\n");
            }
            return;
        };
        if (useShell) {
            this.sshClient.shell(callback);
        }
        else {
            this.sshClient.exec(command !== undefined ? command : "", callback);
        }
        return deferred.promise;
    }
}
exports.ShellExecutor = ShellExecutor;
