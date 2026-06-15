import axios from 'axios';

// The base URL is handled by the Vite proxy, so we can use relative paths
const API = axios.create({
  baseURL: '',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiClient = {
  // ── Predictions ───────────────────────────────────────────────────
  
  /**
   * Predict all 12 properties for a single SMILES string.
   */
  async predictSingle(smiles, threshold = 0.5) {
    const response = await API.post('/api/predict/predict', {
      smiles,
      threshold,
    });
    return response.data;
  },

  /**
   * Upload a CSV file for batch prediction.
   */
  async createBatchJob(file, threshold = 0.5) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold.toString());

    const response = await API.post('/api/predict/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Poll status of a batch prediction job.
   */
  async getBatchJobStatus(jobId) {
    const response = await API.get(`/api/predict/batch/${jobId}`);
    return response.data;
  },

  /**
   * Get preview of completed batch job.
   */
  async getBatchJobPreview(jobId) {
    const response = await API.get(`/api/predict/batch/${jobId}/preview`);
    return response.data;
  },

  /**
   * Download batch predictions results URL.
   */
  getBatchJobDownloadUrl(jobId) {
    return `/api/predict/batch/${jobId}/download`;
  },

  // ── Training ──────────────────────────────────────────────────────

  /**
   * Start a training run in the background.
   */
  async startTraining(config) {
    const payload = {
      run_name: config.runName,
      gin_layers: config.ginLayers,
      hidden_dim: config.hiddenDim,
      dropout: config.dropout,
      pooling: config.pooling,
      dataset: config.dataset,
      split_strategy: config.split,
      batch_size: config.batchSize,
      max_epochs: config.maxEpochs,
      learning_rate: config.lr,
      weight_decay: config.weightDecay,
      early_stopping_patience: config.patience,
      use_pcgrad: config.usePCGrad,
      use_uncertainty_weighting: config.useUncertainty,
    };
    const response = await API.post('/api/train/start', payload);
    return response.data;
  },

  /**
   * List all training runs.
   */
  async getTrainingRuns() {
    const response = await API.get('/api/train/runs');
    return response.data;
  },

  /**
   * Get training status of a single run.
   */
  async getTrainingRun(runId) {
    const response = await API.get(`/api/train/run/${runId}`);
    return response.data;
  },

  /**
   * Get full epoch metrics history for a training run.
   */
  async getTrainingMetrics(runId) {
    const response = await API.get(`/api/train/run/${runId}/metrics`);
    return response.data;
  },

  /**
   * Stop an active training run.
   */
  async stopTrainingRun(runId) {
    const response = await API.post(`/api/train/run/${runId}/stop`);
    return response.data;
  },

  // ── Model Registry ────────────────────────────────────────────────

  /**
   * Get active model details and performance.
   */
  async getModelInfo() {
    const response = await API.get('/api/model/info');
    return response.data;
  },

  /**
   * Get model metrics history.
   */
  async getModelMetrics() {
    const response = await API.get('/api/model/metrics');
    return response.data;
  }
};
