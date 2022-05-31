"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LinuxCommands = void 0;
const osCommands_1 = require("../osCommands");
class LinuxCommands extends osCommands_1.OsCommands {
    getScriptExt() {
        return "sh";
    }
    generateStartScript(workingDirectory, trialJobId, experimentId, trialSequenceId, isMultiPhase, jobIdFileName, command, nniManagerAddress, nniManagerPort, nniManagerVersion, logCollection, exitCodeFile, codeDir, cudaVisibleSetting) {
        return `#!/bin/bash
            export NNI_PLATFORM=remote NNI_SYS_DIR=${workingDirectory} NNI_OUTPUT_DIR=${workingDirectory} NNI_TRIAL_JOB_ID=${trialJobId} \
            NNI_EXP_ID=${experimentId} NNI_TRIAL_SEQ_ID=${trialSequenceId} NNI_CODE_DIR=${codeDir}
            export MULTI_PHASE=${isMultiPhase}
            mkdir -p $NNI_SYS_DIR/code
            cp -r $NNI_CODE_DIR/. $NNI_SYS_DIR/code
            sh $NNI_SYS_DIR/install_nni.sh
            cd $NNI_SYS_DIR/code
            python3 -m nni.tools.trial_tool.trial_keeper --trial_command '${cudaVisibleSetting} ${command}' --nnimanager_ip '${nniManagerAddress}' \
                --nnimanager_port '${nniManagerPort}' --nni_manager_version '${nniManagerVersion}' \
                --job_id_file ${jobIdFileName} \
                --log_collection '${logCollection}' 1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr
            echo $? \`date +%s%3N\` >${exitCodeFile}`;
    }
    generateGpuStatsScript(scriptFolder) {
        return `echo $$ > ${scriptFolder}/pid ; METRIC_OUTPUT_DIR=${scriptFolder} python3 -m nni.tools.gpu_tool.gpu_metrics_collector`;
    }
    createFolder(folderName, sharedFolder = false) {
        let command;
        if (sharedFolder) {
            command = `umask 0; mkdir -p '${folderName}'`;
        }
        else {
            command = `mkdir -p '${folderName}'`;
        }
        return command;
    }
    allowPermission(isRecursive = false, ...folders) {
        const folderString = folders.join("' '");
        let command;
        if (isRecursive) {
            command = `chmod 777 -R '${folderString}'`;
        }
        else {
            command = `chmod 777 '${folderString}'`;
        }
        return command;
    }
    removeFolder(folderName, isRecursive = false, isForce = true) {
        let flags = '';
        if (isForce || isRecursive) {
            flags = `-${isRecursive ? 'r' : 'd'}${isForce ? 'f' : ''} `;
        }
        const command = `rm ${flags}'${folderName}'`;
        return command;
    }
    removeFiles(folderName, filePattern) {
        const files = this.joinPath(folderName, filePattern);
        const command = `rm '${files}'`;
        return command;
    }
    readLastLines(fileName, lineCount = 1) {
        const command = `tail -n ${lineCount} '${fileName}'`;
        return command;
    }
    isProcessAliveCommand(pidFileName) {
        const command = `kill -0 \`cat '${pidFileName}'\``;
        return command;
    }
    isProcessAliveProcessOutput(commandResult) {
        let result = true;
        if (commandResult.exitCode !== 0) {
            result = false;
        }
        return result;
    }
    killChildProcesses(pidFileName, killSelf) {
        let command = `list_descendants ()
                {
                local children=$(ps -o pid= --ppid "$1")

                for pid in $children
                do
                    list_descendants "$pid"
                done

                echo "$children"
                }
            kill $(list_descendants \`cat '${pidFileName}'\`)`;
        if (killSelf) {
            command += `\nkill \`cat '${pidFileName}'\``;
        }
        return command;
    }
    extractFile(tarFileName, targetFolder) {
        const command = `tar -oxzf '${tarFileName}' -C '${targetFolder}'`;
        return command;
    }
    executeScript(script, isFile) {
        let command;
        if (isFile) {
            command = `bash '${script}'`;
        }
        else {
            script = script.replace(/"/g, '\\"');
            const result = script.match(/[^\\]\\\\"/g);
            if (result) {
                result.forEach((res) => {
                    script = script.replace(res, res.replace(/"$/g, '\\"'));
                });
            }
            command = `bash -c "${script}"`;
        }
        return command;
    }
    setPythonPath(pythonPath, command) {
        if (command === undefined || command === '' || pythonPath === undefined || pythonPath === '') {
            return command;
        }
        else {
            return `export PATH=${pythonPath}:$PATH && ${command}`;
        }
    }
    fileExistCommand(filePath) {
        return `test -e ${filePath} && echo True || echo False`;
    }
    getCurrentPath() {
        return `pwd`;
    }
}
exports.LinuxCommands = LinuxCommands;
