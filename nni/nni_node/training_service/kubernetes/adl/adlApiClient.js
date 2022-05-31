"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AdlClientV1 = exports.GeneralK8sClient = exports.AdlClientFactory = void 0;
const fs_1 = __importDefault(require("fs"));
const kubernetesApiClient_1 = require("../kubernetesApiClient");
Object.defineProperty(exports, "GeneralK8sClient", { enumerable: true, get: function () { return kubernetesApiClient_1.GeneralK8sClient; } });
class AdlClientV1 extends kubernetesApiClient_1.KubernetesCRDClient {
    namespace;
    constructor(namespace) {
        super();
        this.namespace = namespace;
        this.crdSchema = JSON.parse(fs_1.default.readFileSync('./config/adl/adaptdl-crd-v1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }
    get operator() {
        return this.client.apis['adaptdl.petuum.com'].v1.namespaces(this.namespace).adaptdljobs;
    }
    get containerName() {
        return 'main';
    }
    async getKubernetesPods(jobName) {
        let result;
        const response = await this.client.api.v1.namespaces(this.namespace).pods
            .get({ qs: { labelSelector: `adaptdl/job=${jobName}` } });
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(response.body);
        }
        else {
            result = Promise.reject(`AdlClient getKubernetesPods failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
}
exports.AdlClientV1 = AdlClientV1;
class AdlClientFactory {
    static createClient(namespace) {
        return new AdlClientV1(namespace);
    }
}
exports.AdlClientFactory = AdlClientFactory;
