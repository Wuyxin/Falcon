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
exports.UnitTestHelpers = exports.RestServer = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const path_1 = __importDefault(require("path"));
const express_1 = __importStar(require("express"));
const http_proxy_1 = __importDefault(require("http-proxy"));
const ts_deferred_1 = require("ts-deferred");
const component_1 = require("common/component");
const log_1 = require("common/log");
const utils_1 = require("common/utils");
const restHandler_1 = require("./restHandler");
let RestServer = class RestServer {
    port;
    urlPrefix;
    server = null;
    logger = log_1.getLogger('RestServer');
    constructor(port, urlPrefix) {
        strict_1.default(!urlPrefix.startsWith('/') && !urlPrefix.endsWith('/'));
        this.port = port;
        this.urlPrefix = urlPrefix;
    }
    start() {
        this.logger.info(`Starting REST server at port ${this.port}, URL prefix: "/${this.urlPrefix}"`);
        const app = express_1.default();
        app.use('/' + this.urlPrefix, rootRouter(this.shutdown.bind(this)));
        app.all('*', (_req, res) => { res.status(404).send(`Outside prefix "/${this.urlPrefix}"`); });
        this.server = app.listen(this.port);
        const deferred = new ts_deferred_1.Deferred();
        this.server.on('listening', () => {
            if (this.port === 0) {
                this.port = this.server.address().port;
            }
            this.logger.info('REST server started.');
            deferred.resolve();
        });
        this.server.on('error', (error) => {
            this.logger.error('REST server error:', error);
            deferred.reject(error);
        });
        return deferred.promise;
    }
    shutdown() {
        this.logger.info('Stopping REST server.');
        if (this.server === null) {
            this.logger.warning('REST server is not running.');
            return Promise.resolve();
        }
        const deferred = new ts_deferred_1.Deferred();
        this.server.close(() => {
            this.logger.info('REST server stopped.');
            deferred.resolve();
        });
        this.server.on('error', (error) => {
            this.logger.error('REST server error:', error);
            deferred.resolve();
        });
        return deferred.promise;
    }
};
RestServer = __decorate([
    component_1.Singleton,
    __metadata("design:paramtypes", [Number, String])
], RestServer);
exports.RestServer = RestServer;
function rootRouter(stopCallback) {
    const router = express_1.Router();
    router.use(express_1.default.json({ limit: '50mb' }));
    router.use('/api/v1/nni', restHandler_1.createRestHandler(stopCallback));
    const logRouter = express_1.Router();
    logRouter.get('*', express_1.default.static(logDirectory ?? utils_1.getLogDir()));
    router.use('/logs', logRouter);
    router.use('/netron', netronProxy());
    router.get('*', express_1.default.static(webuiPath));
    router.get('*', (_req, res) => { res.sendFile(path_1.default.join(webuiPath, 'index.html')); });
    router.all('*', (_req, res) => { res.status(404).send('Not Found'); });
    return router;
}
function netronProxy() {
    const router = express_1.Router();
    const proxy = http_proxy_1.default.createProxyServer();
    router.all('*', (req, res) => {
        delete req.headers.host;
        proxy.web(req, res, { changeOrigin: true, target: netronUrl });
    });
    return router;
}
let webuiPath = path_1.default.resolve('static');
let netronUrl = 'https://netron.app';
let logDirectory = undefined;
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function getPort(server) {
        return server.port;
    }
    UnitTestHelpers.getPort = getPort;
    function setWebuiPath(mockPath) {
        webuiPath = path_1.default.resolve(mockPath);
    }
    UnitTestHelpers.setWebuiPath = setWebuiPath;
    function setNetronUrl(mockUrl) {
        netronUrl = mockUrl;
    }
    UnitTestHelpers.setNetronUrl = setNetronUrl;
    function setLogDirectory(path) {
        logDirectory = path;
    }
    UnitTestHelpers.setLogDirectory = setLogDirectory;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
