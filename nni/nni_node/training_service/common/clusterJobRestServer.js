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
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ClusterJobRestServer = void 0;
const assert_1 = __importDefault(require("assert"));
const body_parser_1 = __importDefault(require("body-parser"));
const express_1 = require("express");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const typescript_string_operations_1 = require("typescript-string-operations");
const component = __importStar(require("common/component"));
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const restServer_1 = require("common/restServer");
const utils_1 = require("common/utils");
let ClusterJobRestServer = class ClusterJobRestServer extends restServer_1.LegacyRestServer {
    API_ROOT_URL = '/api/v1/nni-pai';
    NNI_METRICS_PATTERN = `NNISDK_MEb'(?<metrics>.*?)'`;
    expId = experimentStartupInfo_1.getExperimentId();
    enableVersionCheck = true;
    versionCheckSuccess;
    errorMessage;
    constructor() {
        super();
        const basePort = experimentStartupInfo_1.getBasePort();
        assert_1.default(basePort !== undefined && basePort > 1024);
        this.port = basePort + 1;
    }
    get apiRootUrl() {
        return this.API_ROOT_URL;
    }
    get clusterRestServerPort() {
        if (this.port === undefined) {
            throw new Error('PAI Rest server port is undefined');
        }
        return this.port;
    }
    get getErrorMessage() {
        return this.errorMessage;
    }
    set setEnableVersionCheck(versionCheck) {
        this.enableVersionCheck = versionCheck;
    }
    registerRestHandler() {
        this.app.use(body_parser_1.default.json());
        this.app.use(this.API_ROOT_URL, this.createRestHandler());
    }
    createRestHandler() {
        const router = express_1.Router();
        router.use((req, res, next) => {
            this.log.info(`${req.method}: ${req.url}: body:`, req.body);
            res.setHeader('Content-Type', 'application/json');
            next();
        });
        router.post(`/version/${this.expId}/:trialId`, (req, res) => {
            if (this.enableVersionCheck) {
                try {
                    const checkResultSuccess = req.body.tag === 'VCSuccess' ? true : false;
                    if (this.versionCheckSuccess !== undefined && this.versionCheckSuccess !== checkResultSuccess) {
                        this.errorMessage = 'Version check error, version check result is inconsistent!';
                        this.log.error(this.errorMessage);
                    }
                    else if (checkResultSuccess) {
                        this.log.info(`Version check in trialKeeper success!`);
                        this.versionCheckSuccess = true;
                    }
                    else {
                        this.versionCheckSuccess = false;
                        this.errorMessage = req.body.msg;
                    }
                }
                catch (err) {
                    this.log.error(`json parse metrics error: ${err}`);
                    res.status(500);
                    res.send(err.message);
                }
            }
            else {
                this.log.info(`Skipping version check!`);
            }
            res.send();
        });
        router.post(`/update-metrics/${this.expId}/:trialId`, (req, res) => {
            try {
                this.log.info(`Get update-metrics request, trial job id is ${req.params['trialId']}`);
                this.log.info('update-metrics body is', req.body);
                this.handleTrialMetrics(req.body.jobId, req.body.metrics);
                res.send();
            }
            catch (err) {
                this.log.error(`json parse metrics error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });
        router.post(`/stdout/${this.expId}/:trialId`, (req, res) => {
            if (this.enableVersionCheck && (this.versionCheckSuccess === undefined || !this.versionCheckSuccess)
                && this.errorMessage === undefined) {
                this.errorMessage = `Version check failed, didn't get version check response from trialKeeper,`
                    + ` please check your NNI version in NNIManager and TrialKeeper!`;
            }
            const trialLogDir = path_1.default.join(utils_1.getExperimentRootDir(), 'trials', req.params['trialId']);
            utils_1.mkDirPSync(trialLogDir);
            const trialLogPath = path_1.default.join(trialLogDir, 'stdout_log_collection.log');
            try {
                let skipLogging = false;
                if (req.body.tag === 'trial' && req.body.msg !== undefined) {
                    const metricsContent = req.body.msg.match(this.NNI_METRICS_PATTERN);
                    if (metricsContent && metricsContent.groups) {
                        const key = 'metrics';
                        this.handleTrialMetrics(req.params['trialId'], [metricsContent.groups[key]]);
                        skipLogging = true;
                    }
                }
                if (!skipLogging) {
                    const writeStream = fs_1.default.createWriteStream(trialLogPath, {
                        flags: 'a+',
                        encoding: 'utf8',
                        autoClose: true
                    });
                    writeStream.write(typescript_string_operations_1.String.Format('{0}\n', req.body.msg));
                    writeStream.end();
                }
                res.send();
            }
            catch (err) {
                this.log.error(`json parse stdout data error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });
        return router;
    }
};
ClusterJobRestServer = __decorate([
    component.Singleton,
    __metadata("design:paramtypes", [])
], ClusterJobRestServer);
exports.ClusterJobRestServer = ClusterJobRestServer;
