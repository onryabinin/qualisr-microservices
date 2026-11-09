const $ = id => document.getElementById(id);
const statusLabels = {
  created: 'Создан', queued: 'В очереди', running: 'Выполняется',
  succeeded: 'Завершён', failed: 'Ошибка',
};
const modelLabels = {randomforest: 'Random Forest', xgb: 'XGBoost', catboost: 'CatBoost'};
const modeLabels = {split: 'Групповой split', cv: 'Групповая CV'};
const active = new Set(['queued', 'running']);
const datasets = new Map();
const history = new Map();
const selectedRuns = new Set();
let shown;
let comparisonRequest = 0;

function message(id, text, error = false) {
  $(id).textContent = text;
  $(id).classList.toggle('error', error);
}

async function api(path, options, plainText = false) {
  const response = await fetch('/api' + path, options);
  if (!response.ok) {
    const isJSON = response.headers.get('content-type')?.includes('application/json');
    const detail = isJSON ? (await response.json()).detail : await response.text();
    const reason = detail?.error || detail;
    const text = Array.isArray(reason)
      ? reason.map(item => [item.loc.join('.'), item.msg].join(' · ')).join('\n')
      : typeof reason === 'string' ? reason : JSON.stringify(reason);
    const error = new Error(detail?.import_id ? `${text} · импорт ${detail.import_id}` : text);
    error.status = response.status;
    throw error;
  }
  return plainText ? response.text() : response.json();
}

function node(tag, value) {
  const element = document.createElement(tag);
  element.textContent = value;
  return element;
}

function choice(value, label, checked = false) {
  const input = document.createElement('input');
  Object.assign(input, {type: 'checkbox', value, checked});
  const wrapper = document.createElement('label');
  wrapper.append(input, ' ', label);
  return wrapper;
}

function checked(id) {
  return [...$(id).querySelectorAll('input:checked:not(:disabled)')].map(input => input.value);
}

function syncReferences() {
  const fr = checked('features').includes('fr');
  $('references').hidden = !fr;
  $('references').disabled = !fr;
}

function renderDataset() {
  const dataset = datasets.get($('dataset').value);
  $('features').replaceChildren();
  $('feature-columns').replaceChildren();
  $('dataset-info').textContent = '';
  if (dataset) {
    const {n_samples, n_groups, features, files} = dataset;
    $('dataset-info').textContent = `Строк: ${n_samples} · Групп test_case: ${n_groups}`;
    $('n-splits').max = n_groups;
    $('n-splits').value = Math.max(2, Math.min(n_groups, Number($('n-splits').value)));
    for (const [group, columns] of Object.entries(features)) {
      $('features').append(choice(group, group, ['nr', 'fr'].includes(group)));
      $('feature-columns').append(node('dt', group), node('dd', columns.join(', ')));
    }
    $('feature-columns').append(node('dt', 'Файлы'), node('dd', Object.keys(files).join(', ')));
    const refs = [...$('references').querySelectorAll('input')];
    for (const input of refs) {
      input.disabled = !(features.fr || []).some(column => column.toLowerCase().endsWith('_' + input.value));
      input.checked = false;
    }
    const preferred = refs.find(input => !input.disabled);
    if (preferred) preferred.checked = true;
  }
  syncReferences();
}

async function loadDatasets(selected) {
  const items = await api('/datasets');
  datasets.clear();
  for (const item of items) datasets.set(item.id, item);
  $('dataset').replaceChildren(...items.map(item => new Option(item.name, item.id)));
  if (selected) $('dataset').value = selected;
  renderDataset();
}

function renderModels(defaults) {
  for (const [name, params] of Object.entries(defaults)) {
    const wrapper = node('div', '');
    wrapper.className = 'model';
    const toggle = choice(name, modelLabels[name], name === 'randomforest');
    const editor = document.createElement('textarea');
    editor.rows = Object.keys(params).length + 2;
    editor.value = JSON.stringify(params, null, 2);
    editor.disabled = name !== 'randomforest';
    editor.setAttribute('aria-label', `Гиперпараметры ${modelLabels[name]}`);
    toggle.firstChild.onchange = () => { editor.disabled = !toggle.firstChild.checked; };
    wrapper.append(toggle, editor);
    $('models').append(wrapper);
  }
}

function settings() {
  const features = checked('features');
  const frRefs = checked('references');
  if (!features.length) throw new Error('Выберите хотя бы одну группу признаков.');
  if (features.includes('fr') && !frRefs.length) throw new Error('Выберите хотя бы один референс FR.');
  const models = {};
  for (const name of checked('models')) {
    const editor = $('models').querySelector(`[value="${name}"]`).closest('.model').querySelector('textarea');
    try {
      const params = JSON.parse(editor.value);
      if (!params || Array.isArray(params) || typeof params !== 'object') throw new SyntaxError();
      models[name] = params;
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      throw new Error(`Введите гиперпараметры ${modelLabels[name]} как JSON-объект.`);
    }
  }
  if (!Object.keys(models).length) throw new Error('Выберите хотя бы один регрессор.');
  const mode = $('mode').value;
  return {
    dataset_id: $('dataset').value, features, models, mode,
    ...(features.includes('fr') ? {fr_refs: frRefs} : {}),
    seed: $('seed').valueAsNumber, split_seed: $('split-seed').valueAsNumber,
    ...(mode === 'cv' ? {n_splits: $('n-splits').valueAsNumber} : {test_size: 0.2}),
  };
}

function table(target, headers, rows) {
  const grid = document.createElement('table');
  grid.createTHead().insertRow().append(...headers.map(header => {
    const cell = node('th', header);
    cell.scope = 'col';
    return cell;
  }));
  const body = grid.createTBody();
  for (const cells of rows) {
    const row = body.insertRow();
    for (const cell of cells) row.insertCell().append(cell ?? '—');
  }
  $(target).replaceChildren(grid);
}

const correlationHeaders = ['Модель / метрика', 'Источник', 'PLCC', 'SRCC', 'Строк', 'Выборка'];
const number = value => value == null ? '—' : value.toFixed(4);

function correlationCells(row, folds = []) {
  const matching = folds.filter(fold => fold.model === row.model && fold.source === row.source);
  const count = matching.length ? matching.reduce((sum, fold) => sum + fold.n_samples, 0) : row.n_samples;
  const same = matching.length ? matching.every(fold => fold.same_samples_as_regressor) : row.same_samples_as_regressor;
  return [
    modelLabels[row.model] || row.model, row.source === 'regressor' ? 'Регрессор' : 'IQA-метрика',
    number(row.plcc) + (row.plcc_std == null ? '' : ` ± ${number(row.plcc_std)}`),
    number(row.srcc) + (row.srcc_std == null ? '' : ` ± ${number(row.srcc_std)}`),
    count, same ? 'Общая' : 'Неполная · несопоставимо',
  ];
}

function datasetLabel(run) {
  return datasets.get(run.dataset_id)?.name || run.dataset_id;
}

function runModels(run) {
  // Older run records contain only the Random Forest from the original template.
  return Object.entries(run.models ?? {randomforest: {}})
    .filter(([, params]) => params !== null).map(([name]) => name);
}

function runMode(run) {
  return `${modeLabels[run.mode]} · ${run.mode === 'cv' ? `${run.n_splits} фолдов` : '80/20'}`;
}

function renderHistory() {
  const rows = [...history.values()].reverse().map(run => {
    const toggle = choice(run.id, '', selectedRuns.has(run.id)).firstChild;
    toggle.disabled = run.status !== 'succeeded';
    toggle.setAttribute('aria-label', `Выбрать эксперимент ${run.id}`);
    toggle.onchange = () => {
      if (toggle.checked) selectedRuns.add(run.id);
      else selectedRuns.delete(run.id);
      $('compare').disabled = selectedRuns.size < 2;
      $('comparison').hidden = true;
      comparisonRequest++;
    };
    const open = node('button', 'Открыть');
    open.onclick = () => openRun(run.id);
    open.setAttribute('aria-label', `Открыть эксперимент ${run.id}`);
    const rows = run.results?.correlations.filter(row => row.source === 'regressor') || [];
    return [
      toggle, new Date(run.created_at).toLocaleString('ru-RU'), datasetLabel(run),
      run.features.join(', '), runMode(run), `${run.seed} / ${run.split_seed}`,
      statusLabels[run.status],
      rows.length ? rows.map(row => `${modelLabels[row.model]} ${number(row.plcc)} / ${number(row.srcc)}`).join(' · ')
        : runModels(run).map(name => modelLabels[name]).join(', '),
      open,
    ];
  });
  table('history', ['Выбор', 'Дата', 'Датасет', 'Признаки', 'Режим', 'Seeds', 'Статус', 'Регрессоры · PLCC / SRCC', 'Детали'], rows);
  if (!rows.length) message('history-status', 'Запусков пока нет.');
  $('compare').disabled = selectedRuns.size < 2;
}

async function loadHistory() {
  try {
    const runs = await api('/experiments');
    history.clear();
    for (const run of runs) history.set(run.id, run);
    for (const id of selectedRuns) if (!history.has(id)) selectedRuns.delete(id);
    message('history-status', '');
    renderHistory();
  } catch (error) {
    message('history-status', error.message, true);
  }
}

function artifactURL(artifact) {
  return `/artifacts/${artifact.id}`;
}

function artifactLink(artifact, label = artifact.path) {
  const link = node('a', label);
  link.href = '/api' + artifactURL(artifact);
  return link;
}

function links(id, artifacts, fullPath = false) {
  const items = artifacts.map(file => {
    const item = document.createElement('li');
    item.append(artifactLink(file, fullPath ? file.path : file.name));
    return item;
  });
  $(id).replaceChildren(...items);
}

function predictionFile(run, artifacts, model) {
  const file = artifacts.find(item => item.path === `${run.output_path}/predictions/predictions_${model}.csv`);
  if (!file) throw new Error(`Отсутствует CSV предсказаний для ${modelLabels[model]}.`);
  return file;
}

function renderImportance(view, model) {
  const {run, artifacts} = view;
  const plots = artifacts.filter(file => {
    const relative = file.path.slice(run.output_path.length + 1);
    return file.path.startsWith(run.output_path + '/') &&
      /^(fold_\d+\/)?importances\//.test(relative) && file.name === `importance_${model}.png`;
  });
  $('importance').replaceChildren();
  for (const file of plots) {
    const panel = document.createElement('details');
    const fold = file.path.match(/\/fold_(\d+)\//);
    panel.open = !fold;
    const image = document.createElement('img');
    image.src = '/api' + artifactURL(file);
    image.alt = `Важность признаков · ${modelLabels[model]}`;
    image.loading = 'lazy';
    const csv = artifacts.find(item => item.path === file.path.replace(/\.png$/, '.csv'));
    panel.append(node('summary', fold ? `Фолд ${Number(fold[1])}` : modelLabels[model]), image);
    if (csv) panel.append(artifactLink(csv, 'Скачать CSV важности'));
    $('importance').append(panel);
  }
  if (!plots.length) $('importance').textContent = 'Графики важности для этого регрессора отсутствуют.';
}

async function renderScatter(view) {
  const model = $('prediction-model').value;
  renderImportance(view, model);
  if (window.Plotly) window.Plotly.purge($('scatter'));
  message('scatter-status', 'Загрузка предсказаний…');
  try {
    const file = predictionFile(view.run, view.artifacts, model);
    const rows = await api(artifactURL(file) + '/predictions');
    if (shown !== view || model !== $('prediction-model').value) return;
    if (!window.Plotly) throw new Error('Не удалось загрузить Plotly.');
    const plot = {
      type: 'scatter', mode: 'markers', x: rows.map(row => row.mos), y: rows.map(row => row.prediction),
      text: rows.map(row => row.sample_id.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')),
      customdata: rows.map(row => row.fold),
      hovertemplate: '%{text}<br>MOS: %{x:.3f}<br>Предсказание: %{y:.3f}' +
        (view.run.mode === 'cv' ? '<br>Фолд: %{customdata}' : '') + '<extra></extra>',
    };
    await window.Plotly.newPlot($('scatter'), [plot], {
      xaxis: {title: {text: 'MOS'}}, yaxis: {title: {text: 'Предсказанный MOS'}},
      height: 400, margin: {l: 65, r: 25, t: 20, b: 55},
    }, {responsive: true, displaylogo: false});
    if (shown === view && model === $('prediction-model').value) {
      message('scatter-status', `${rows.length} ${view.run.mode === 'cv' ? 'OOF-предсказаний' : 'тестовых предсказаний'}`);
    }
  } catch (error) {
    if (shown === view && model === $('prediction-model').value) message('scatter-status', error.message, true);
  }
}

function renderResults(view) {
  const {run} = view;
  const results = run.results;
  if (!results) return;
  const folds = results.fold_correlations || [];
  const cv = run.mode === 'cv';
  $('evaluation').textContent = cv
    ? 'Среднее ± выборочное стандартное отклонение по фолдам. Число строк суммировано по фолдам. Scatter показывает OOF-предсказания.'
    : 'Корреляции на общей тестовой выборке. Строки с неполным покрытием отмечены в таблице.';
  table('correlations', correlationHeaders, results.correlations.map(row => correlationCells(row, folds)));
  $('fold-panel').hidden = !cv;
  if (cv) table('folds', ['Фолд', ...correlationHeaders], folds.map(row => [row.fold, ...correlationCells(row)]));
  $('prediction-model').replaceChildren(...runModels(run).map(name => new Option(modelLabels[name], name)));
  $('results').hidden = false;
  void renderScatter(view);
}

async function loadLog(view) {
  try {
    const log = await api(`/experiments/${view.id}/log`, undefined, true);
    if (shown === view) message('log', log || 'Журнал пока пуст.');
  } catch (error) {
    if (shown === view) message('log', error.status === 404 ? 'Журнал ещё недоступен.' : error.message, error.status !== 404);
  }
}

async function pollRun(view, run) {
  try {
    while (shown === view) {
      view.run = run || await api(`/experiments/${view.id}`);
      if (shown !== view) return;
      run = view.run;
      const changed = history.get(run.id)?.status !== run.status;
      history.set(run.id, run);
      if (changed) renderHistory();
      const parts = [datasetLabel(run), runMode(run), run.features.join(', '),
        `seed ${run.seed}`, `split seed ${run.split_seed}`, run.id];
      $('run-meta').textContent = parts.join(' · ');
      message('run-status', [statusLabels[run.status], run.error].filter(Boolean).join(' · '), Boolean(run.error));
      $('start-created').hidden = run.status !== 'created';
      $('log-panel').hidden = run.status === 'created';
      if (run.status === 'created') return;
      if ($('log-panel').open) void loadLog(view);
      if (!active.has(run.status)) break;
      await new Promise(resolve => setTimeout(resolve, 1000));
      run = undefined;
    }
    if (shown !== view) return;
    view.artifacts = await api(`/experiments/${view.id}/artifacts`);
    if (shown !== view) return;
    const mainFiles = view.artifacts.filter(file =>
      ['config.json', 'runner.log'].includes(file.path) ||
      (file.path.startsWith(run.output_path + '/') &&
        /^(correlations\/[^/]+\.(csv|png)|predictions\/[^/]+\.csv|metadata\/cross_validation_folds\.csv)$/
          .test(file.path.slice(run.output_path.length + 1))));
    links('exports', mainFiles, true);
    links('artifacts', view.artifacts, true);
    $('downloads').hidden = !view.artifacts.length;
    renderResults(view);
  } catch (error) {
    if (shown === view) message('run-status', error.message, true);
  }
}

function openRun(id, run) {
  const view = {id};
  shown = view;
  $('run-details').hidden = false;
  $('results').hidden = true;
  $('downloads').hidden = true;
  $('start-created').hidden = true;
  $('log-panel').hidden = true;
  $('fold-panel').open = false;
  $('log-panel').open = false;
  $('log').textContent = '';
  $('run-meta').textContent = id;
  message('run-status', 'Загрузка эксперимента…');
  $('run-details').scrollIntoView();
  void pollRun(view, run);
}

async function evaluationKey(run) {
  const artifacts = await api(`/experiments/${run.id}/artifacts`);
  const file = predictionFile(run, artifacts, runModels(run)[0]);
  const rows = await api(artifactURL(file) + '/predictions');
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row.fold)) groups.set(row.fold, []);
    groups.get(row.fold).push(row.sample_id);
  }
  const partition = [...groups.values()].map(ids => JSON.stringify(ids.sort())).sort();
  return JSON.stringify([run.dataset_id, run.mode, partition]);
}

async function compare() {
  const request = ++comparisonRequest;
  $('comparison').hidden = false;
  $('comparison-table').replaceChildren();
  message('comparison-status', 'Проверяю совпадение оценочных sample_id и фолдов…');
  const runs = [...selectedRuns].map(id => history.get(id));
  try {
    const keys = await Promise.all(runs.map(evaluationKey));
    if (request !== comparisonRequest) return;
    const cohorts = [...new Set(keys)];
    message('comparison-status', cohorts.length === 1
      ? 'У регрессоров одинаковые оценочные sample_id и разбиение. Для CV показано среднее по фолдам.'
      : 'Запуски из разных групп несопоставимы. Различается датасет, протокол или разбиение. Для CV показано среднее по фолдам.',
      cohorts.length > 1);
    const rows = runs.flatMap((run, index) => run.results.correlations.map(row => [
      run.id, datasetLabel(run), runMode(run), cohorts.indexOf(keys[index]) + 1,
      ...correlationCells(row, run.results.fold_correlations),
    ]));
    table('comparison-table', ['Запуск', 'Датасет', 'Режим', 'Группа выборки', ...correlationHeaders], rows);
  } catch (error) {
    if (request === comparisonRequest) message('comparison-status', error.message, true);
  }
}

$('dataset').onchange = renderDataset;
$('features').onchange = syncReferences;
$('mode').onchange = () => {
  const cv = $('mode').value === 'cv';
  $('split-setting').hidden = $('split-note').hidden = cv;
  $('cv-setting').hidden = $('n-splits').disabled = !cv;
};
$('import-form').onsubmit = async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector('button');
  button.disabled = true;
  message('import-status', 'Импортирую датасет…');
  try {
    const result = await api('/datasets/import', {method: 'POST', body: new FormData(form)});
    await loadDatasets(result.dataset.id);
    form.reset();
    message('import-status', `Импортирован ${result.dataset.name} · ${result.import_id}`);
  } catch (error) {
    message('import-status', error.message, true);
  } finally {
    button.disabled = false;
  }
};
$('experiment-form').onsubmit = async event => {
  event.preventDefault();
  $('run-button').disabled = true;
  message('submit-status', 'Создаю эксперимент…');
  try {
    const run = await api('/experiments', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(settings()),
    });
    message('submit-status', `Эксперимент поставлен в очередь · ${run.id}`);
    openRun(run.id, run);
  } catch (error) {
    message('submit-status', error.message, true);
  } finally {
    $('run-button').disabled = false;
  }
};
$('log-panel').ontoggle = () => { if ($('log-panel').open) void loadLog(shown); };
$('start-created').onclick = async () => {
  const view = shown;
  const button = $('start-created');
  button.disabled = true;
  try {
    const run = await api(`/experiments/${view.id}/run`, {method: 'POST'});
    if (shown === view) void pollRun(view, run);
  } catch (error) {
    if (shown === view) {
      if (error.status === 409) void pollRun(view);
      else message('run-status', error.message, true);
    }
  } finally {
    button.disabled = false;
  }
};
$('refresh-log').onclick = () => loadLog(shown);
$('prediction-model').onchange = () => renderScatter(shown);
$('refresh-history').onclick = loadHistory;
$('compare').onclick = compare;

try {
  const [, defaults] = await Promise.all([loadDatasets(), api('/model-defaults')]);
  renderModels(defaults);
  $('run-button').disabled = !datasets.size;
} catch (error) {
  message('submit-status', error.message, true);
}
await loadHistory();
