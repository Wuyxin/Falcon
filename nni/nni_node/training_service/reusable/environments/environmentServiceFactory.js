"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createEnvironmentService = void 0;
const amlEnvironmentService_1 = require("./amlEnvironmentService");
const openPaiEnvironmentService_1 = require("./openPaiEnvironmentService");
const localEnvironmentService_1 = require("./localEnvironmentService");
const remoteEnvironmentService_1 = require("./remoteEnvironmentService");
const kubeflowEnvironmentService_1 = require("./kubernetes/kubeflowEnvironmentService");
const frameworkcontrollerEnvironmentService_1 = require("./kubernetes/frameworkcontrollerEnvironmentService");
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const nniConfig_1 = require("common/nniConfig");
const utils_1 = require("common/utils");
const dlcEnvironmentService_1 = require("./dlcEnvironmentService");
async function createEnvironmentService(config) {
    const info = experimentStartupInfo_1.ExperimentStartupInfo.getInstance();
    const configAsAny = config;
    switch (config.platform) {
        case 'local':
            return new localEnvironmentService_1.LocalEnvironmentService(configAsAny, info);
        case 'remote':
            return new remoteEnvironmentService_1.RemoteEnvironmentService(configAsAny, info);
        case 'aml':
            return new amlEnvironmentService_1.AMLEnvironmentService(configAsAny, info);
        case 'openpai':
            return new openPaiEnvironmentService_1.OpenPaiEnvironmentService(configAsAny, info);
        case 'kubeflow':
            return new kubeflowEnvironmentService_1.KubeflowEnvironmentService(configAsAny, info);
        case 'frameworkcontroller':
            return new frameworkcontrollerEnvironmentService_1.FrameworkControllerEnvironmentService(configAsAny, info);
        case 'dlc':
            return new dlcEnvironmentService_1.DlcEnvironmentService(configAsAny, info);
    }
    const esConfig = await nniConfig_1.getCustomEnvironmentServiceConfig(config.platform);
    if (esConfig === null) {
        throw new Error(`${config.platform} is not a supported training service!`);
    }
    const esModule = utils_1.importModule(esConfig.nodeModulePath);
    const esClass = esModule[esConfig.nodeClassName];
    return new esClass(configAsAny, info);
}
exports.createEnvironmentService = createEnvironmentService;
