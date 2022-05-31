"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.kubernetesScriptFormat = exports.KubernetesTrialJobDetail = void 0;
class KubernetesTrialJobDetail {
    id;
    status;
    message;
    submitTime;
    startTime;
    endTime;
    tags;
    url;
    workingDirectory;
    form;
    kubernetesJobName;
    queryJobFailedCount;
    constructor(id, status, submitTime, workingDirectory, form, kubernetesJobName, url) {
        this.id = id;
        this.status = status;
        this.message = 'Pending for creating the trial job.';
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.kubernetesJobName = kubernetesJobName;
        this.tags = [];
        this.queryJobFailedCount = 0;
        this.url = url;
    }
}
exports.KubernetesTrialJobDetail = KubernetesTrialJobDetail;
exports.kubernetesScriptFormat = `#!/bin/bash
export NNI_PLATFORM={0}
export NNI_SYS_DIR={1}
export NNI_OUTPUT_DIR={2}
export MULTI_PHASE=false
export NNI_TRIAL_JOB_ID={3}
export NNI_EXP_ID={4}
export NNI_CODE_DIR={5}
export NNI_TRIAL_SEQ_ID={6}
{7}
mkdir -p $NNI_SYS_DIR/code
mkdir -p $NNI_OUTPUT_DIR
cp -r $NNI_CODE_DIR/. $NNI_SYS_DIR/code
sh $NNI_SYS_DIR/install_nni.sh
cd $NNI_SYS_DIR/code
python3 -m nni.tools.trial_tool.trial_keeper --trial_command '{8}' --nnimanager_ip {9} --nnimanager_port {10} \
--nni_manager_version '{11}' --log_collection '{12}' 1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr`;
