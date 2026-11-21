// Pure JavaScript contract checks with mock DOM/canvas; no browser or network.
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.resolve(root,process.argv[2]||'assets/operator_showcase_v2/explorer.html'),'utf8');
assert(!html.includes('__FIELD_DATA__'));
assert(!/<(?:script|link)[^>]+(?:src|href)=/i.test(html),'Demo must remain offline');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length,1);
let paintCalls=0;
const context={fillRect(){paintCalls++;},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillText(){}};
const nodes={};
function node(id){
  if(!nodes[id])nodes[id]={value:id==='mode'?'prediction':id==='time'?'49':'0',textContent:'',children:[],events:{},
    appendChild(v){this.children.push(v);},addEventListener(k,v){this.events[k]=v;},getContext(){return context;}};
  return nodes[id];
}
let frame;
const sandbox={document:{getElementById:node,createElement(){return {value:'',textContent:''};}},requestAnimationFrame(f){frame=f;}};
vm.createContext(sandbox);
new vm.Script(scripts[0][1]).runInContext(sandbox,{timeout:10000});
assert.equal(nodes.segment.children.length,6);
assert.match(nodes.smaRMSE.textContent,/^\d+\.\d{3} K$/);
assert.equal(nodes.timeLabel.textContent,'1.00');
const first=nodes.smaRMSE.textContent;
nodes.segment.value='5';nodes.segment.events.input();
assert.notEqual(nodes.smaRMSE.textContent,first);
nodes.mode.value='error';nodes.mode.events.input();
assert.equal(nodes.smaLo.textContent,'0.00 K');
nodes.time.value='0';nodes.time.events.input();assert.equal(nodes.timeLabel.textContent,'0.00');
nodes.play.onclick();frame(200);assert.equal(+nodes.time.value,1);assert.match(nodes.play.textContent,/Pause/);
nodes.play.onclick();const paused=nodes.time.value;frame(500);assert.equal(nodes.time.value,paused);
nodes.reset.onclick();assert.equal(+nodes.segment.value,0);assert.equal(+nodes.time.value,49);assert.equal(nodes.mode.value,'prediction');
assert(paintCalls>10000);
console.log(JSON.stringify({status:'passed_offline_explorer_logic_contracts',segments:6,
  checks:['embedded_data','no_external_resources','statistics','segment_change','error_view','time_slider','play_pause','reset'],
  real_browser_rendering_verified:false,scope:'mock DOM/canvas; browser file navigation was denied by local browser policy'}));
