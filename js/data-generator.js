/* ============================================
   Data Generator for Binary Classification
   ============================================ */

const DataGenerator = {
  /**
   * Generate random dataset for binary classification
   * @param {number} numFeatures - Number of input features
   * @param {number} numSamples - Number of samples (default 10)
   * @returns {Object} { features: [[]], labels: [], headers: [] }
   */
  generate(numFeatures, numSamples = 10) {
    const features = [];
    const labels = [];
    const headers = [];

    // Create headers
    for (let i = 0; i < numFeatures; i++) {
      headers.push(`X${i + 1}`);
    }
    headers.push('Y');

    // Generate data
    for (let i = 0; i < numSamples; i++) {
      const row = [];
      let sum = 0;
      for (let j = 0; j < numFeatures; j++) {
        const val = parseFloat((Math.random()).toFixed(4));
        row.push(val);
        sum += val;
      }
      features.push(row);
      // Simple threshold for label generation
      const avg = sum / numFeatures;
      labels.push(avg > 0.5 ? 1 : 0);
    }

    // Ensure we have both classes
    const hasZero = labels.includes(0);
    const hasOne = labels.includes(1);
    if (!hasZero && labels.length > 1) labels[0] = 0;
    if (!hasOne && labels.length > 1) labels[labels.length - 1] = 1;

    return { features, labels, headers };
  },

  /**
   * Render data table to HTML
   * @param {Object} data - Generated data object
   * @returns {string} HTML table string
   */
  renderTable(data) {
    const { features, labels, headers } = data;

    let html = '<table class="data-table" id="dataTable">';

    // Header
    html += '<thead><tr>';
    html += '<th>#</th>';
    headers.forEach(h => {
      html += `<th>${h}</th>`;
    });
    html += '</tr></thead>';

    // Body
    html += '<tbody>';
    features.forEach((row, i) => {
      html += `<tr data-row="${i}">`;
      html += `<td style="color: var(--text-muted)">${i + 1}</td>`;
      row.forEach(val => {
        html += `<td>${val.toFixed(4)}</td>`;
      });
      const label = labels[i];
      html += `<td class="label-${label}">${label}</td>`;
      html += '</tr>';
    });
    html += '</tbody></table>';

    return html;
  }
};
