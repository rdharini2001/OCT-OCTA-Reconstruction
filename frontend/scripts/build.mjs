import { mkdir, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
const root=resolve(import.meta.dirname,'..');
await mkdir(resolve(root,'dist'),{recursive:true});
for (const file of ['index.html']) await copyFile(resolve(root,file),resolve(root,'dist',file));
for (const file of ['app.js','styles.css']) await copyFile(resolve(root,'src',file),resolve(root,'dist',file));
console.log('OCTAVE static frontend generated (no network dependencies).');
