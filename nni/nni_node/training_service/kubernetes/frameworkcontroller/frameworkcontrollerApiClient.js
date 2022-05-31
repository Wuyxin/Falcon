"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GeneralK8sClient = exports.FrameworkControllerClientFactory = void 0;
const fs_1 = __importDefault(require("fs"));
const kubernetesApiClient_1 = require("../kubernetesApiClient");
Object.defineProperty(exports, "GeneralK8sClient", { enumerable: true, get: function () { return kubernetesApiClient_1.GeneralK8sClient; } });
class FrameworkControllerClientV1 extends kubernetesApiClient_1.KubernetesCRDClient {
    namespace;
    constructor(namespace) {
        super();
        this.namespace = namespace ? namespace : "default";
        this.crdSchema = JSON.parse(fs_1.default.readFileSync('./config/frameworkcontroller/frameworkcontrollerjob-crd-v1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }
    get operator() {
        return this.client.apis['frameworkcontroller.microsoft.com'].v1.namespaces(this.namespace).frameworks;
    }
    get containerName() {
        return 'framework';
    }
}
class FrameworkControllerClientFactory {
    static createClient(namespace) {
        return new FrameworkControllerClientV1(namespace);
    }
}
exports.FrameworkControllerClientFactory = FrameworkControllerClientFactory;
