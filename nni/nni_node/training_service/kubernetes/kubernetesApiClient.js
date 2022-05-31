"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GeneralK8sClient = exports.KubernetesCRDClient = void 0;
const kubernetes_client_1 = require("kubernetes-client");
const log_1 = require("common/log");
function getKubernetesConfig() {
    if ('KUBERNETES_SERVICE_HOST' in process.env) {
        return kubernetes_client_1.config.getInCluster();
    }
    else {
        return kubernetes_client_1.config.fromKubeconfig();
    }
}
class GeneralK8sClient {
    client;
    log = log_1.getLogger('GeneralK8sClient');
    namespace = 'default';
    constructor() {
        this.client = new kubernetes_client_1.Client1_10({ config: getKubernetesConfig(), version: '1.9' });
        this.client.loadSpec();
    }
    set setNamespace(namespace) {
        this.namespace = namespace;
    }
    get getNamespace() {
        return this.namespace;
    }
    matchStorageClass(response) {
        const adlSupportedProvisioners = [
            new RegExp("microk8s.io/hostpath"),
            new RegExp(".*cephfs.csi.ceph.com"),
            new RegExp(".*azure.*"),
            new RegExp("\\b" + "efs" + "\\b")
        ];
        const templateLen = adlSupportedProvisioners.length, responseLen = response.items.length;
        let i = 0, j = 0;
        for (; i < responseLen; i++) {
            const provisioner = response.items[i].provisioner;
            for (; j < templateLen; j++) {
                if (provisioner.match(adlSupportedProvisioners[j])) {
                    return response.items[i].metadata.name;
                }
            }
        }
        return "Not Found!";
    }
    async getStorageClass() {
        let result;
        const response = await this.client.apis["storage.k8s.io"].v1beta1.storageclasses.get();
        const storageClassType = this.matchStorageClass(response.body);
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            if (storageClassType != "Not Found!") {
                result = Promise.resolve(storageClassType);
            }
            else {
                result = Promise.reject("No StorageClasses are supported!");
            }
        }
        else {
            result = Promise.reject(`List storageclasses failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
    async createDeployment(deploymentManifest) {
        let result;
        const response = await this.client.apis.apps.v1.namespaces(this.namespace)
            .deployments.post({ body: deploymentManifest });
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(response.body.metadata.uid);
        }
        else {
            result = Promise.reject(`Create deployment failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
    async deleteDeployment(deploymentName) {
        let result;
        const response = await this.client.apis.apps.v1.namespaces(this.namespace)
            .deployment(deploymentName).delete();
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(true);
        }
        else {
            result = Promise.reject(`Delete deployment failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
    async createConfigMap(configMapManifest) {
        let result;
        const response = await this.client.api.v1.namespaces(this.namespace)
            .configmaps.post({ body: configMapManifest });
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(true);
        }
        else {
            result = Promise.reject(`Create configMap failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
    async createPersistentVolumeClaim(pvcManifest) {
        let result;
        const response = await this.client.api.v1.namespaces(this.namespace)
            .persistentvolumeclaims.post({ body: pvcManifest });
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(true);
        }
        else {
            result = Promise.reject(`Create pvc failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
    async createSecret(secretManifest) {
        let result;
        const response = await this.client.api.v1.namespaces(this.namespace)
            .secrets.post({ body: secretManifest });
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(true);
        }
        else {
            result = Promise.reject(`Create secrets failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
}
exports.GeneralK8sClient = GeneralK8sClient;
class KubernetesCRDClient {
    client;
    log = log_1.getLogger('KubernetesCRDClient');
    crdSchema;
    namespace = 'default';
    constructor() {
        this.client = new kubernetes_client_1.Client1_10({ config: getKubernetesConfig() });
        this.client.loadSpec();
    }
    get jobKind() {
        if (this.crdSchema
            && this.crdSchema.spec
            && this.crdSchema.spec.names
            && this.crdSchema.spec.names.kind) {
            return this.crdSchema.spec.names.kind;
        }
        else {
            throw new Error('KubeflowOperatorClient: getJobKind failed, kind is undefined in crd schema!');
        }
    }
    get apiVersion() {
        if (this.crdSchema
            && this.crdSchema.spec
            && this.crdSchema.spec.version) {
            return this.crdSchema.spec.version;
        }
        else {
            throw new Error('KubeflowOperatorClient: get apiVersion failed, version is undefined in crd schema!');
        }
    }
    async createKubernetesJob(jobManifest) {
        let result;
        const response = await this.operator.post({ body: jobManifest });
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(true);
        }
        else {
            result = Promise.reject(`KubernetesApiClient createKubernetesJob failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
    async getKubernetesJob(kubeflowJobName) {
        let result;
        const response = await this.operator(kubeflowJobName)
            .get();
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(response.body);
        }
        else {
            result = Promise.reject(`KubernetesApiClient getKubernetesJob failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
    async deleteKubernetesJob(labels) {
        let result;
        const matchQuery = Array.from(labels.keys())
            .map((labelKey) => `${labelKey}=${labels.get(labelKey)}`)
            .join(',');
        try {
            const deleteResult = await this.operator()
                .delete({
                qs: {
                    labelSelector: matchQuery,
                    propagationPolicy: 'Background'
                }
            });
            if (deleteResult.statusCode && deleteResult.statusCode >= 200 && deleteResult.statusCode <= 299) {
                result = Promise.resolve(true);
            }
            else {
                result = Promise.reject(`KubernetesApiClient, delete labels ${matchQuery} get wrong statusCode ${deleteResult.statusCode}`);
            }
        }
        catch (err) {
            result = Promise.reject(err);
        }
        return result;
    }
}
exports.KubernetesCRDClient = KubernetesCRDClient;
