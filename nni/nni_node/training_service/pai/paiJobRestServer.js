"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.PAIJobRestServer = void 0;
const clusterJobRestServer_1 = require("../common/clusterJobRestServer");
class PAIJobRestServer extends clusterJobRestServer_1.ClusterJobRestServer {
    parameterFileMetaList = [];
    paiTrainingService;
    constructor(paiTrainingService) {
        super();
        this.paiTrainingService = paiTrainingService;
    }
    handleTrialMetrics(jobId, metrics) {
        for (const singleMetric of metrics) {
            this.paiTrainingService.MetricsEmitter.emit('metric', {
                id: jobId,
                data: singleMetric
            });
        }
    }
    createRestHandler() {
        const router = super.createRestHandler();
        router.post(`/parameter-file-meta`, (req, res) => {
            try {
                this.log.info('POST /parameter-file-meta, body is', req.body);
                this.parameterFileMetaList.push(req.body);
                res.send();
            }
            catch (err) {
                this.log.error(`POST parameter-file-meta error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });
        router.get(`/parameter-file-meta`, (_req, res) => {
            try {
                this.log.info(`GET /parameter-file-meta`);
                res.send(this.parameterFileMetaList);
            }
            catch (err) {
                this.log.error(`GET parameter-file-meta error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });
        return router;
    }
}
exports.PAIJobRestServer = PAIJobRestServer;
