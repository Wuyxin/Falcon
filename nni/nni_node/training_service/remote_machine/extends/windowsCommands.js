"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WindowsCommands = void 0;
const osCommands_1 = require("../osCommands");
class WindowsCommands extends osCommands_1.OsCommands {
    pathSpliter = '\\';
    getScriptExt() {
        return "cmd";
    }
    generateStartScript(workingDirectory, trialJobId, experimentId, trialSequenceId, isMultiPhase, jobIdFileName, command, nniManagerAddress, nniManagerPort, nniManagerVersion, logCollection, exitCodeFile, codeDir, cudaVisibleSetting) {
        return `echo off
            set NNI_PLATFORM=remote
            set NNI_SYS_DIR=${workingDirectory}
            set NNI_OUTPUT_DIR=${workingDirectory}
            set NNI_TRIAL_JOB_ID=${trialJobId}
            set NNI_EXP_ID=${experimentId}
            set NNI_TRIAL_SEQ_ID=${trialSequenceId}
            set MULTI_PHASE=${isMultiPhase}
            set NNI_CODE_DIR=${codeDir}
            ${cudaVisibleSetting !== "" ? "set " + cudaVisibleSetting : ""}
            md %NNI_SYS_DIR%/code
            robocopy /s %NNI_CODE_DIR%/. %NNI_SYS_DIR%/code
            cd %NNI_SYS_DIR%/code
            python -c "import nni" 2>nul
            if not %ERRORLEVEL% EQU 0 (
                echo installing NNI as exit code of "import nni" is %ERRORLEVEL%
                python -m pip install --user --upgrade nni
            )

            echo starting script
            python -m nni.tools.trial_tool.trial_keeper --trial_command "${command}" --nnimanager_ip "${nniManagerAddress}" --nnimanager_port "${nniManagerPort}" --nni_manager_version "${nniManagerVersion}" --log_collection "${logCollection}" --job_id_file ${jobIdFileName} 1>%NNI_OUTPUT_DIR%/trialkeeper_stdout 2>%NNI_OUTPUT_DIR%/trialkeeper_stderr

            echo save exit code(%ERRORLEVEL%) and time
            echo|set /p="%ERRORLEVEL% " > ${exitCodeFile}
            powershell -command "Write (((New-TimeSpan -Start (Get-Date "01/01/1970") -End (Get-Date).ToUniversalTime()).TotalMilliseconds).ToString("0")) | Out-file ${exitCodeFile} -Append -NoNewline -encoding utf8"`;
    }
    generateGpuStatsScript(scriptFolder) {
        return `powershell -command $env:Path=If($env:prePath){$env:prePath}Else{$env:Path};$env:METRIC_OUTPUT_DIR='${scriptFolder}';$app = Start-Process -FilePath python -NoNewWindow -passthru -ArgumentList '-m nni.tools.gpu_tool.gpu_metrics_collector' -RedirectStandardOutput ${scriptFolder}\\scriptstdout -RedirectStandardError ${scriptFolder}\\scriptstderr;Write $PID ^| Out-File ${scriptFolder}\\pid -NoNewline -encoding utf8;wait-process $app.ID`;
    }
    createFolder(folderName, sharedFolder = false) {
        let command;
        if (sharedFolder) {
            command = `mkdir "${folderName}"\r\nICACLS "${folderName}" /grant "Users":F`;
        }
        else {
            command = `mkdir "${folderName}"`;
        }
        return command;
    }
    allowPermission(isRecursive = false, ...folders) {
        let commands = "";
        folders.forEach(folder => {
            commands += `ICACLS "${folder}" /grant "Users":F${isRecursive ? " /T" : ""}\r\n`;
        });
        return commands;
    }
    removeFolder(folderName, isRecursive = false, isForce = true) {
        let flags = '';
        if (isForce || isRecursive) {
            flags = `${isRecursive ? ' /s' : ''}${isForce ? ' /q' : ''}`;
        }
        const command = `rmdir${flags} "${folderName}"`;
        return command;
    }
    removeFiles(folderName, filePattern) {
        const files = this.joinPath(folderName, filePattern);
        const command = `del "${files}"`;
        return command;
    }
    readLastLines(fileName, lineCount = 1) {
        const command = `powershell.exe Get-Content "${fileName}" -Tail ${lineCount}`;
        return command;
    }
    isProcessAliveCommand(pidFileName) {
        const command = `powershell.exe Get-Process -Id (get-content "${pidFileName}") -ErrorAction SilentlyContinue`;
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
        let command = `powershell "$ppid=(type ${pidFileName}); function Kill-Tree {Param([int]$subppid);` +
            `Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $subppid } | ForEach-Object { Kill-Tree $_.ProcessId }; ` +
            `if ($subppid -ne $ppid){Stop-Process -Id $subppid -Force"}}` +
            `kill-tree $ppid"`;
        if (killSelf) {
            command += `;Stop-Process -Id $ppid`;
        }
        return command;
    }
    extractFile(tarFileName, targetFolder) {
        const command = `tar -xf "${tarFileName}" -C "${targetFolder}"`;
        return command;
    }
    executeScript(script, _isFile) {
        const command = `${script}`;
        return command;
    }
    setPythonPath(pythonPath, command) {
        if (command === undefined || command === '' || pythonPath === undefined || pythonPath === '') {
            return command;
        }
        else {
            return `set path=${pythonPath};%path% && set prePath=%path% && ${command}`;
        }
    }
    fileExistCommand(filePath) {
        return `powershell Test-Path ${filePath} -PathType Leaf`;
    }
    getCurrentPath() {
        return `chdir`;
    }
}
exports.WindowsCommands = WindowsCommands;
