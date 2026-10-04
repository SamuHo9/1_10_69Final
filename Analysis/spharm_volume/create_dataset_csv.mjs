import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { Workbook } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const splitRoot = path.join(root, 'SplitData');
const output = path.join(splitRoot, 'dataset_details.csv');
const readJson = async p => JSON.parse(await fs.readFile(p, 'utf8'));
const exists = async p => { try { await fs.access(p); return true; } catch(e) { if(e.code==='ENOENT') return false; throw e; } };
assert(!await exists(output), 'dataset_details.csv already exists; refusing to overwrite.');
const manifest = await readJson(path.join(splitRoot, 'split_manifest.json'));
const people = new Map(manifest.map(r => [r.subject_id, r]));
assert.equal(people.size, 381);
const records = [];
for(const side of ['left','right']) {
  const sourceFolder = path.join(root, 'SPHARM', `SPHARM_${side.toUpperCase()}`);
  const statusFiles = (await fs.readdir(sourceFolder)).filter(n => /^spharm_status_shard\d+\.json$/.test(n)).sort();
  for(const statusFile of statusFiles) {
    const status = await readJson(path.join(sourceFolder, statusFile));
    for(const item of status.subjects) {
      const inputName = path.win32.basename(item.input);
      const base = inputName.replace(/\.nii\.gz$/, '');
      const id = base.match(/sub-([^_]+)/)[1];
      const person = people.get(id);
      assert(person, `No split assignment for ${id}`);
      const hemisphereLabel = base.match(/_(Healthy|TLE)_/)[1];
      assert.equal(hemisphereLabel, person[`${side}_filename_label`]);
      assert.equal(item.success, person[`${side}_available`]);
      const meshName = `${base}_SPHARM.vtk`;
      const relativeFolder = `${person.split}/SPHARM_${side.toUpperCase()}/spharm_results`;
      const meshPath = path.join(splitRoot, relativeFolder, meshName);
      const included = await exists(meshPath);
      const coefficients = await exists(path.join(splitRoot,relativeFolder,`${base}_SPHARM.coef`));
      const grid = await exists(path.join(splitRoot,relativeFolder,`${base}_SPHARM_grid.vtk`));
      assert.equal(included,item.success, `Mismatch between status and copied surface: ${id} ${side}`);
      assert.equal(coefficients,item.success);
      assert.equal(grid,item.success);
      records.push({
        subject_id: `sub-${id}`,
        split: person.split,
        side,
        dataset_partition: `${person.split}_${side}`,
        spharm_status: item.success ? 'passed' : 'failed',
        included_in_split: included ? 1 : 0,
        hemisphere_label_original: hemisphereLabel,
        person_group_inferred: person.inferred_person_group,
        cohort_inferred: person.inferred_cohort,
        spharm_coefficients_available: coefficients ? 1 : 0,
        spharm_grid_available: grid ? 1 : 0,
        input_filename: inputName,
        spharm_mesh_filename: meshName,
        split_mesh_path: included ? `${relativeFolder}/${meshName}` : '',
        status_source: `SPHARM/SPHARM_${side.toUpperCase()}/${statusFile}`,
      });
    }
  }
}
records.sort((a,b) => (a.split==='train'?0:1)-(b.split==='train'?0:1) || a.side.localeCompare(b.side) || a.subject_id.localeCompare(b.subject_id));
assert.equal(records.length,762);
assert.equal(new Set(records.map(r=>`${r.subject_id}/${r.side}`)).size,762);
assert.equal(records.filter(r=>r.spharm_status==='passed').length,750);
assert.equal(records.filter(r=>r.spharm_status==='failed').length,12);
const headers = Object.keys(records[0]);
const matrix = [headers,...records.map(r=>headers.map(h=>r[h]))];
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('Dataset details');
sheet.getRange(`A1:O${matrix.length}`).values = matrix;
sheet.getRange(`A1:O${matrix.length}`).format.font = { name:'Arial', size:10 };
sheet.getRange('A1:O1').format.font = {name:'Arial',size:10,bold:true,color:'#FFFFFF'};
sheet.getRange('A1:O1').format.fill = '#334155';
sheet.getRange('A1:O1').format.rowHeight = 28;
sheet.getRange(`A1:F${matrix.length}`).format.columnWidth = 24;
sheet.freezePanes.freezeRows(1);
workbook.recalculate();
const inspected = await workbook.inspect({kind:'table',range:"'Dataset details'!A1:F6",include:'values',tableMaxRows:6,tableMaxCols:6,maxChars:1800});
console.log(inspected.ndjson);
const preview = await workbook.render({sheetName:'Dataset details',range:'A1:F12',scale:1,format:'png'});
await fs.writeFile(path.join(here,'dataset_details_preview.png'),new Uint8Array(await preview.arrayBuffer()));
// CSV is the requested deliverable; serialize the documented value getter with RFC 4180 escaping.
const savedValues = sheet.getRange(`A1:O${matrix.length}`).values;
assert.deepEqual(savedValues,matrix);
const escapeCsv = value => {
  const s = value === null || value === undefined ? '' : String(value);
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"','""')}"` : s;
};
await fs.writeFile(output,'\uFEFF'+savedValues.map(row=>row.map(escapeCsv).join(',')).join('\r\n')+'\r\n',{encoding:'utf8',flag:'wx'});
const counts = {};
for(const record of records) {
  const key = `${record.dataset_partition}/${record.spharm_status}`;
  counts[key] = (counts[key] || 0)+1;
}
console.log(JSON.stringify({output,rows:records.length,counts},null,2));
