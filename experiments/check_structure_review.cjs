/* Local browser smoke test. Uses an installed Playwright module, no network install.
 * SONGVIZ_PLAYWRIGHT_MODULE=/path/to/playwright node experiments/check_structure_review.cjs URL
 * Optional third arg: template path to test a draft template without editing a saved review.
 */
const fs = require('node:fs');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.SONGVIZ_PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const url = process.argv[2];
  assert(url, 'Pass the localhost review URL');
  const browser = await chromium.launch({headless:true, args:['--autoplay-policy=no-user-gesture-required']});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    const errors = [];
    const audioRequests = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('request', request => {
      if (/\/original\.wav(?:\?.*)?$/.test(request.url())) {
        audioRequests.push({resourceType:request.resourceType(),range:request.headers().range || ''});
      }
    });
    if (process.argv[3]) {
      await page.route(url, async route => {
        const response = await route.fetch();
        const html = await response.text();
        const data = html.match(/<script id="review-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
        const hash = html.match(/const manifestHash = '([a-f0-9]+)'/)[1];
        await route.fulfill({response,body:fs.readFileSync(process.argv[3],'utf8').replace('{{REVIEW_JSON}}',data).replace('{{MANIFEST_SHA}}',hash)});
      });
    }
    await page.goto(url);
    await page.waitForFunction(() => state.metadataReady && audio.readyState >= 2);
    assert.equal(audioRequests.some(request=>request.resourceType==='fetch'),false,'Audio must not be fetched into a full Blob before media loading');
    assert.equal(audioRequests.some(request=>!request.range),false,'Audio must not issue an un-ranged full-file request');
    assert.equal(audioRequests.some(request=>request.range),true,'Native audio request did not include a Range header');
    await page.evaluate(() => {audio.muted=true;});
    assert.equal(await page.locator('#boundary-questions article').count(),4);
    assert.equal(await page.locator('#repeat-questions article').count(),3);
    assert.equal(await page.locator('select').evaluateAll(xs=>xs.every(x=>x.value==='unreviewed')),true);
    await page.locator('#add-mark').click();
    assert.equal(await page.locator('.marked-item').count(),0, 'Blank marker must not become zero');
    await page.locator('#marker-time').fill('42.5');
    await page.locator('#add-mark').click();
    await page.locator('.marked-item textarea').fill('Browser test observation');
    await page.locator('#boundary-questions select').first().selectOption('within_section');
    await page.locator('#boundary-0-corrected').fill('77.1');
    await page.locator('#boundary-0-notes').fill('Browser test boundary note');
    await page.locator('#repeat-questions select').first().selectOption('variation');
    await page.getByRole('button',{name:'Play A',exact:true}).first().click();
    await page.waitForFunction(() => !audio.paused && !audio.seeking);
    assert.equal(await page.evaluate(()=>Math.abs(audio.currentTime-review.repeat_questions[0].a_start_s)<1),true,'Playback did not seek to passage A');
    assert.notEqual(await page.evaluate(()=>state.stopAt),null, 'Programmatic seek lost bounded stop');
    await page.evaluate(() => seekTo(100));
    assert.equal(await page.evaluate(()=>state.stopAt),null, 'User seek should cancel bounded stop');
    await page.evaluate(() => playSegment(30,30.4));
    await page.waitForFunction(()=>audio.paused && audio.currentTime>=30.4,{},{timeout:5000});
    assert.equal(await page.locator('#boundary-0-notes').inputValue(),'Browser test boundary note');
    await page.locator('#boundary-0-corrected').fill('-1');
    await page.locator('#export').click();
    assert.match(await page.locator('#export-status').innerText(),/Fix/);
    await page.locator('#boundary-0-corrected').fill('77.1');
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#export').click();
    const download = await downloadEvent;
    const feedback = JSON.parse(fs.readFileSync(await download.path(),'utf8'));
    assert.equal(download.suggestedFilename(),'songviz-structure-feedback.json');
    assert.match(feedback.manifest_sha256,/^[a-f0-9]{64}$/);
    assert.equal(feedback.boundary_answers[0].answer,'within_section');
    assert.equal(feedback.boundary_answers[0].corrected_time_s,77.1);
    assert.equal(typeof feedback.boundary_answers[0].time_s,'number');
    assert.equal(feedback.repeat_answers[0].answer,'variation');
    assert.equal(feedback.marked_observations[0].time_s,42.5);
    const actualManifest = await (await page.request.get(new URL('manifest.json',url).href)).body();
    assert.equal(feedback.manifest_sha256,require('node:crypto').createHash('sha256').update(actualManifest).digest('hex'));
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true,'Mobile horizontal overflow');
    await page.route(/\/original\.wav(?:\?.*)?$/,route=>route.abort());
    await page.evaluate(() => loadMetadata(true));
    await page.locator('#retry-audio').waitFor({state:'visible'});
    assert.equal(await page.evaluate(()=>state.metadataReady),false);
    assert.equal(await page.locator('#boundary-questions select').first().inputValue(),'within_section');
    assert.equal(await page.locator('#repeat-questions select').first().inputValue(),'variation');
    assert.equal(await page.locator('#boundary-0-notes').isEnabled(),true);
    await page.locator('#boundary-0-notes').fill('Note written while audio unavailable');
    await page.unroute(/\/original\.wav(?:\?.*)?$/);
    await page.locator('#retry-audio').click();
    await page.waitForFunction(()=>state.metadataReady);
    assert.equal(await page.locator('#boundary-0-notes').inputValue(),'Note written while audio unavailable');
    assert.deepEqual(errors,[]);
    console.log('PASS: metadata, controls, bounded playback, user seek, notes persistence, validation, export/hash, mobile layout, audio error/retry.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exit(1);});
