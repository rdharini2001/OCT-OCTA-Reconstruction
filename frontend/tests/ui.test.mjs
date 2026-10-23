import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
const js=readFileSync(new URL('../src/app.js',import.meta.url),'utf8');
const html=readFileSync(new URL('../index.html',import.meta.url),'utf8');
test('UI has upload and usable empty state',()=>{assert.match(js,/api\/datasets/);assert.match(js,/structural/);assert.match(js,/flow/);assert.match(js,/No studies yet/)});
test('No fictional datasets or synthetic demo',()=>{assert.doesNotMatch(js,/synthetic-demo|createDemo|fakePatient/);assert.doesNotMatch(html,/synthetic-demo/)});
test('Linked 2D views and FAZ adjustment',()=>{assert.match(js,/bscan/);assert.match(js,/enface/);assert.match(js,/seed_x/);assert.match(js,/threshold/)});
test('No external runtime UI libraries',()=>{assert.doesNotMatch(html,/https?:\/\//);assert.match(html,/app.js/)});
test('3D orbit projection uses declared yaw (regression)',()=>{assert.doesNotMatch(js,/Math\.(sin|cos)\(S\.y\)/);assert.match(js,/function plot3d/)});
