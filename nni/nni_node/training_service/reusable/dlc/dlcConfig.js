"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DlcEnvironmentInformation = exports.DlcTrialConfig = exports.DlcClusterConfig = void 0;
const trialConfig_1 = require("training_service/common/trialConfig");
const environment_1 = require("../environment");
class DlcClusterConfig {
    type;
    image;
    podCount;
    ecsSpec;
    constructor(type, image, podCount, ecsSpec) {
        this.type = type;
        this.image = image;
        this.podCount = podCount;
        this.ecsSpec = ecsSpec;
    }
}
exports.DlcClusterConfig = DlcClusterConfig;
class DlcTrialConfig extends trialConfig_1.TrialConfig {
    image;
    command;
    codeDir;
    constructor(codeDir, command, image) {
        super("", codeDir, 0);
        this.codeDir = codeDir;
        this.command = command;
        this.image = image;
    }
}
exports.DlcTrialConfig = DlcTrialConfig;
class DlcEnvironmentInformation extends environment_1.EnvironmentInformation {
    dlcClient;
    currentMessageIndex = -1;
}
exports.DlcEnvironmentInformation = DlcEnvironmentInformation;
