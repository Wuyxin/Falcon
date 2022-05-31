"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
require("app-module-path/register");
const fs_1 = __importDefault(require("fs"));
const typescript_ioc_1 = require("typescript-ioc");
const component = __importStar(require("common/component"));
const datastore_1 = require("common/datastore");
const experimentManager_1 = require("common/experimentManager");
const arguments_1 = require("common/globals/arguments");
const log_1 = require("common/log");
const manager_1 = require("common/manager");
const tensorboardManager_1 = require("common/tensorboardManager");
const nniDataStore_1 = require("core/nniDataStore");
const nniExperimentsManager_1 = require("core/nniExperimentsManager");
const nniTensorboardManager_1 = require("core/nniTensorboardManager");
const nnimanager_1 = require("core/nnimanager");
const sqlDatabase_1 = require("core/sqlDatabase");
const rest_server_1 = require("rest_server");
const path_1 = __importDefault(require("path"));
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const args = arguments_1.parseArgs(process.argv.slice(2));
async function start() {
    log_1.getLogger('main').info('Start NNI manager');
    typescript_ioc_1.Container.bind(manager_1.Manager).to(nnimanager_1.NNIManager).scope(typescript_ioc_1.Scope.Singleton);
    typescript_ioc_1.Container.bind(datastore_1.Database).to(sqlDatabase_1.SqlDB).scope(typescript_ioc_1.Scope.Singleton);
    typescript_ioc_1.Container.bind(datastore_1.DataStore).to(nniDataStore_1.NNIDataStore).scope(typescript_ioc_1.Scope.Singleton);
    typescript_ioc_1.Container.bind(experimentManager_1.ExperimentManager).to(nniExperimentsManager_1.NNIExperimentsManager).scope(typescript_ioc_1.Scope.Singleton);
    typescript_ioc_1.Container.bind(tensorboardManager_1.TensorboardManager).to(nniTensorboardManager_1.NNITensorboardManager).scope(typescript_ioc_1.Scope.Singleton);
    const ds = component.get(datastore_1.DataStore);
    await ds.init();
    const restServer = new rest_server_1.RestServer(args.port, args.urlPrefix);
    await restServer.start();
}
function shutdown() {
    component.get(manager_1.Manager).stopExperiment();
}
process.on('SIGTERM', shutdown);
process.on('SIGBREAK', shutdown);
process.on('SIGINT', shutdown);
experimentStartupInfo_1.setExperimentStartupInfo(args);
const logDirectory = path_1.default.join(args.experimentsDirectory, args.experimentId, 'log');
fs_1.default.mkdirSync(logDirectory, { recursive: true });
log_1.startLogging(path_1.default.join(logDirectory, 'nnimanager.log'));
log_1.setLogLevel(args.logLevel);
start().then(() => {
    log_1.getLogger('main').debug('start() returned.');
}).catch((error) => {
    try {
        log_1.getLogger('main').error('Failed to start:', error);
    }
    catch (loggerError) {
        console.log('Failed to start:', error);
        console.log('Seems logger is faulty:', loggerError);
    }
    process.exit(1);
});
