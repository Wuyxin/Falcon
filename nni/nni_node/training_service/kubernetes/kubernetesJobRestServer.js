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
Object.defineProperty(exports, "__esModule", { value: true });
exports.KubernetesJobRestServer = void 0;
const typescript_ioc_1 = require("typescript-ioc");
const component = __importStar(require("common/component"));
const clusterJobRestServer_1 = require("../common/clusterJobRestServer");
const kubernetesTrainingService_1 = require("./kubernetesTrainingService");
let KubernetesJobRestServer = class KubernetesJobRestServer extends clusterJobRestServer_1.ClusterJobRestServer {
    kubernetesTrainingService;
    constructor(kubernetesTrainingService) {
        super();
        this.kubernetesTrainingService = kubernetesTrainingService;
    }
    handleTrialMetrics(jobId, metrics) {
        if (this.kubernetesTrainingService === undefined) {
            throw Error('kubernetesTrainingService not initialized!');
        }
        for (const singleMetric of metrics) {
            this.kubernetesTrainingService.MetricsEmitter.emit('metric', {
                id: jobId,
                data: singleMetric
            });
        }
    }
};
__decorate([
    typescript_ioc_1.Inject,
    __metadata("design:type", kubernetesTrainingService_1.KubernetesTrainingService)
], KubernetesJobRestServer.prototype, "kubernetesTrainingService", void 0);
KubernetesJobRestServer = __decorate([
    component.Singleton,
    __metadata("design:paramtypes", [kubernetesTrainingService_1.KubernetesTrainingService])
], KubernetesJobRestServer);
exports.KubernetesJobRestServer = KubernetesJobRestServer;
