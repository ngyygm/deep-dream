/* ==========================================
   Shared concept/detail UI
   ========================================== */

window.ConceptDetail = (function () {
  'use strict';

  function ensureStyles() {
    if (document.getElementById('concept-detail-shared-styles')) return;
    const style = document.createElement('style');
    style.id = 'concept-detail-shared-styles';
    style.textContent = `
      .dd-concept-grid{display:grid;grid-template-columns:120px minmax(0,1fr);gap:.45rem .75rem;font-size:.84rem;margin-bottom:.9rem}
      .dd-concept-grid span{color:var(--text-muted)}
      .dd-concept-grid strong{font-weight:500;overflow-wrap:anywhere}
      .dd-concept-section{border-top:1px solid var(--border-color);padding-top:.8rem;margin-top:.8rem}
      .dd-concept-actions{display:flex;gap:.45rem;flex-wrap:wrap;justify-content:flex-end;margin-top:.9rem}
      .dd-concept-content{border:1px solid var(--border-color);border-radius:8px;padding:.75rem;background:var(--bg-secondary);max-height:36vh;overflow:auto;line-height:1.6}
      .dd-concept-pill{display:inline-flex;align-items:center;border:1px solid var(--border-color);border-radius:999px;padding:.12rem .5rem;font-size:.72rem;color:var(--text-muted);margin:.1rem .25rem .1rem 0}
      .dd-concept-evidence{border:1px solid var(--border-color);background:var(--bg-secondary);border-radius:.55rem;padding:.55rem .65rem}
      .dd-concept-diff{margin-top:.5rem;border:1px solid var(--border-color);border-radius:8px;overflow:hidden;background:var(--bg-secondary)}
      .dd-concept-diff-head{display:flex;justify-content:space-between;gap:.5rem;padding:.4rem .55rem;border-bottom:1px solid var(--border-color);font-size:.75rem;color:var(--text-muted)}
      .dd-concept-diff-body{padding:.5rem .6rem;font-size:.8rem;line-height:1.65;max-height:220px;overflow:auto;word-break:break-word}
      .dd-diff-token.added,.dd-diff-line.added{color:#16a34a;text-decoration:none}
      .dd-diff-token.removed,.dd-diff-line.removed{color:#dc2626;text-decoration:line-through}
      .dd-diff-token.unchanged{color:var(--text-primary)}
      .dd-diff-line{display:block;padding:.2rem .3rem;border-radius:4px;margin:.12rem 0}
      .dd-version-row{border:1px solid var(--border-color);border-radius:8px;padding:.65rem;background:var(--bg-secondary);margin-bottom:.55rem}
      .dd-version-full summary{cursor:pointer;font-size:.78rem;color:var(--text-muted);margin-top:.45rem}
      .dd-version-switcher{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap}
      .dd-version-switcher select{flex:1;min-width:160px;height:34px}
    `;
    document.head.appendChild(style);
  }

  function versionId(concept) {
    return concept?.absolute_id || concept?.version_id || concept?.id || '';
  }

  function familyId(concept) {
    return concept?.family_id || '';
  }

  function title(concept) {
    if (!concept) return '-';
    return concept.name || concept.summary || concept.content || concept.relation_type || concept.family_id || concept.version_id || concept.id || '-';
  }

  function versionText(v) {
    return String(v?.content || v?.summary || v?.name || '').trim();
  }

  function endpointInfo(relation) {
    const meta = relation?.metadata || {};
    return {
      entity1VersionId: relation?.entity1_absolute_id || meta.entity1_absolute_id || '',
      entity2VersionId: relation?.entity2_absolute_id || meta.entity2_absolute_id || '',
      entity1FamilyId: relation?.entity1_family_id || meta.entity1_family_id || '',
      entity2FamilyId: relation?.entity2_family_id || meta.entity2_family_id || '',
      entity1Name: relation?.entity1_name || meta.entity1_name || '',
      entity2Name: relation?.entity2_name || meta.entity2_name || '',
    };
  }

  function renderGrid(rows) {
    return `
      <div class="dd-concept-grid">
        ${rows.map(([label, value, mono]) => `
          <span>${escapeHtml(label)}</span>
          <strong class="${mono ? 'mono' : ''}">${value == null || value === '' ? '-' : escapeHtml(String(value))}</strong>
        `).join('')}
      </div>
    `;
  }

  function pill(text, color) {
    return `<span class="dd-concept-pill" style="${color ? `border-color:${color};color:${color};` : ''}">${escapeHtml(text || '-')}</span>`;
  }

  function renderRoleBadge(role) {
    const isRelation = role === 'relation';
    return pill(isRelation ? 'Relation' : 'Entity', isRelation ? '#f59e0b' : '#14b8a6');
  }

  function renderEvidenceList(evidence) {
    if (!Array.isArray(evidence) || !evidence.length) return '';
    return `
      <div style="display:flex;flex-direction:column;gap:.45rem;">
        ${evidence.map(ev => `
          <div class="dd-concept-evidence">
            <div style="font-size:.78rem;line-height:1.55;color:var(--text-primary);">${highlightEvidenceSentence(ev)}</div>
            <div style="margin-top:.35rem;display:flex;gap:.25rem;flex-wrap:wrap;">
              ${pill(ev.match_type || 'match', '#38bdf8')}
              ${ev.confidence != null ? pill(`conf ${Number(ev.confidence).toFixed(2)}`, '#22c55e') : ''}
              ${ev.start_offset != null ? pill(`${ev.start_offset}-${ev.end_offset}`, '#94a3b8') : ''}
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  function highlightEvidenceSentence(ev) {
    const sentence = String(ev?.sentence || '');
    const quote = String(ev?.quote || '');
    if (!sentence || !quote) return escapeHtml(sentence || quote);
    const idx = sentence.indexOf(quote);
    if (idx < 0) return escapeHtml(sentence);
    return `${escapeHtml(sentence.slice(0, idx))}<mark style="background:rgba(250,204,21,.35);color:inherit;border-radius:.2rem;padding:0 .08rem;">${escapeHtml(quote)}</mark>${escapeHtml(sentence.slice(idx + quote.length))}`;
  }

  function renderConceptBody(concept, opts) {
    opts = opts || {};
    const role = concept.role === 'relation' ? 'relation' : 'entity';
    const text = versionText(concept);
    const evidenceHtml = role === 'relation' ? '' : renderEvidenceList(opts.evidence || []);
    const endpointsHtml = role === 'relation' ? renderEndpointBox(concept, opts) : '';
    const versionCount = opts.versionCount || concept.version_count || concept.versions_count || 0;

    // Build relevance/rank info row (only when available, e.g. from search results)
    const relevance = typeof concept.relevance === 'number' ? concept.relevance : null;
    const rank = typeof concept._rank === 'number' ? concept._rank : null;
    const hasSearchInfo = relevance !== null || rank !== null;
    const searchInfoHtml = hasSearchInfo ? `
      <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem;">
        ${rank !== null ? pill(`#${rank}`, '#14b8a6') : ''}
        ${relevance !== null ? `<span class="dd-concept-pill" style="border-color:${relevance >= 80 ? '#22c55e' : relevance >= 50 ? '#3b82f6' : relevance >= 30 ? '#f59e0b' : '#94a3b8'};color:${relevance >= 80 ? '#22c55e' : relevance >= 50 ? '#3b82f6' : relevance >= 30 ? '#f59e0b' : '#94a3b8'};font-weight:600;">${Math.round(relevance)}%</span>` : ''}
      </div>
    ` : '';

    return `
      <div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.75rem;">
        ${renderRoleBadge(role)}
        ${versionCount > 1 ? pill(`${versionCount} versions`, '#f59e0b') : ''}
      </div>
      ${searchInfoHtml}
      ${renderGrid([
        ['Family', familyId(concept), true],
        ['Version', versionId(concept), true],
        ['Source', concept.source_document || '', false],
        ['Episode', concept.episode_version_id || '', true],
        ['Processed', formatDateMs(concept.processed_time), false],
        ['Event', formatDate(concept.event_time), false],
      ])}
      ${endpointsHtml}
      ${opts.inlineVersionSwitcher ? `<div class="dd-concept-section" data-concept-version-switcher="${escapeAttr(familyId(concept))}">${spinnerHtml('spinner-sm')} 加载版本切换器...</div>` : ''}
      <div class="dd-concept-section">
        <div style="font-weight:600;margin-bottom:.35rem;">内容</div>
        <div class="dd-concept-content md-content" data-concept-version-content>${renderMarkdown(text || '暂无内容')}</div>
      </div>
      ${evidenceHtml ? `<div class="dd-concept-section" data-concept-version-evidence><div style="font-weight:600;margin-bottom:.35rem;">原文句子</div>${evidenceHtml}</div>` : `<div class="dd-concept-section" data-concept-version-evidence style="display:none;"></div>`}
      ${opts.extraHtml || ''}
      <div class="dd-concept-actions">
        ${opts.moreButton ? `<button class="btn btn-secondary btn-sm" data-concept-action="more">更多详情</button>` : ''}
        <button class="btn btn-secondary btn-sm" data-concept-action="edit"><i data-lucide="pencil" style="width:14px;height:14px;margin-right:4px;"></i>编辑</button>
        <button class="btn btn-primary btn-sm" data-concept-action="versions"><i data-lucide="history" style="width:14px;height:14px;margin-right:4px;"></i>版本历史</button>
        ${opts.focusButton ? `<button class="btn btn-secondary btn-sm" data-concept-action="focus">聚焦模式</button>` : ''}
        ${opts.exitFocusButton ? `<button class="btn btn-secondary btn-sm" data-concept-action="exit-focus">退出聚焦</button>` : ''}
      </div>
    `;
  }

  function renderEndpointBox(relation, opts) {
    const ep = endpointInfo(relation);
    const label1 = ep.entity1Name || opts?.endpointLabel1 || ep.entity1FamilyId || ep.entity1VersionId || '-';
    const label2 = ep.entity2Name || opts?.endpointLabel2 || ep.entity2FamilyId || ep.entity2VersionId || '-';
    return `
      <div class="dd-concept-section" style="border-top:0;padding-top:0;margin-top:0;">
        <div style="font-size:.78rem;color:var(--text-muted);margin-bottom:.35rem;">关系端点</div>
        <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:.55rem;align-items:center;">
          <span data-relation-endpoint="1" class="badge badge-primary" style="white-space:normal;text-align:center;">${escapeHtml(label1)}</span>
          <span style="color:var(--text-muted);">--</span>
          <span data-relation-endpoint="2" class="badge badge-info" style="white-space:normal;text-align:center;">${escapeHtml(label2)}</span>
        </div>
      </div>
    `;
  }

  async function hydrateRelationEndpoints(root, relation, opts) {
    const ep = endpointInfo(relation);
    const resolver = opts?.resolveConceptLabel;
    if (!resolver) return;
    const [label1, label2] = await Promise.all([
      ep.entity1Name ? Promise.resolve(ep.entity1Name) : resolver(ep.entity1FamilyId, ep.entity1VersionId),
      ep.entity2Name ? Promise.resolve(ep.entity2Name) : resolver(ep.entity2FamilyId, ep.entity2VersionId),
    ]);
    const e1 = root.querySelector('[data-relation-endpoint="1"]');
    const e2 = root.querySelector('[data-relation-endpoint="2"]');
    if (e1) e1.textContent = label1 || '-';
    if (e2) e2.textContent = label2 || '-';
  }

  function compareVersionTime(a, b) {
    const ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
    const tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
    if (ta !== tb) return ta - tb;
    return String(versionId(a) || '').localeCompare(String(versionId(b) || ''));
  }

  function tokenizeDiffText(text) {
    return String(text || '').match(/[\r\n]+|./gs) || [];
  }

  function renderVersionDiff(current, previous) {
    const currentText = versionText(current);
    const previousText = previous ? versionText(previous) : '';
    if (!previous) {
      return `
        <div class="dd-concept-diff">
          <div class="dd-concept-diff-head"><span>初始版本</span><span>+${tokenizeDiffText(currentText).filter(t => t.trim()).length}</span></div>
          <div class="dd-concept-diff-body">${currentText ? `<span class="dd-diff-token added">${escapeHtml(currentText)}</span>` : '<span class="dd-diff-line unchanged">空内容</span>'}</div>
        </div>
      `;
    }
    if (currentText === previousText) {
      return `
        <div class="dd-concept-diff">
          <div class="dd-concept-diff-head"><span>相比上一版本</span><span>无内容变化</span></div>
          <div class="dd-concept-diff-body"><span class="dd-diff-token unchanged">${escapeHtml(truncate(currentText || '空内容', 260))}</span></div>
        </div>
      `;
    }
    const diff = buildInlineDiff(previousText, currentText);
    return `
      <div class="dd-concept-diff">
        <div class="dd-concept-diff-head"><span>相比上一版本</span><span><span style="color:#16a34a;">+${diff.added}</span> <span style="color:#dc2626;">-${diff.removed}</span></span></div>
        <div class="dd-concept-diff-body">${diff.html}</div>
      </div>
    `;
  }

  function buildInlineDiff(oldText, newText) {
    const oldTokens = tokenizeDiffText(oldText);
    const newTokens = tokenizeDiffText(newText);
    if (oldTokens.length * newTokens.length > 500000) return buildCoarseDiff(oldText, newText);
    const rows = oldTokens.length + 1;
    const cols = newTokens.length + 1;
    const dp = Array.from({ length: rows }, () => new Uint16Array(cols));
    for (let i = oldTokens.length - 1; i >= 0; i--) {
      for (let j = newTokens.length - 1; j >= 0; j--) {
        dp[i][j] = oldTokens[i] === newTokens[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    const parts = [];
    let added = 0;
    let removed = 0;
    let i = 0;
    let j = 0;
    while (i < oldTokens.length && j < newTokens.length) {
      if (oldTokens[i] === newTokens[j]) {
        parts.push({ type: 'unchanged', text: oldTokens[i] });
        i++;
        j++;
      } else if (dp[i + 1][j] >= dp[i][j + 1]) {
        parts.push({ type: 'removed', text: oldTokens[i] });
        if (oldTokens[i].trim()) removed++;
        i++;
      } else {
        parts.push({ type: 'added', text: newTokens[j] });
        if (newTokens[j].trim()) added++;
        j++;
      }
    }
    while (i < oldTokens.length) {
      parts.push({ type: 'removed', text: oldTokens[i] });
      if (oldTokens[i].trim()) removed++;
      i++;
    }
    while (j < newTokens.length) {
      parts.push({ type: 'added', text: newTokens[j] });
      if (newTokens[j].trim()) added++;
      j++;
    }
    const merged = [];
    parts.forEach(part => {
      const last = merged[merged.length - 1];
      if (last && last.type === part.type) last.text += part.text;
      else merged.push({ ...part });
    });
    return {
      added,
      removed,
      html: merged.map(part => part.text ? `<span class="dd-diff-token ${part.type}">${escapeHtml(part.text)}</span>` : '').join(''),
    };
  }

  function buildCoarseDiff(oldText, newText) {
    const oldLines = String(oldText || '').split(/\n+/).filter(Boolean);
    const newLines = String(newText || '').split(/\n+/).filter(Boolean);
    return {
      added: newLines.length,
      removed: oldLines.length,
      html: `
        ${oldLines.map(line => `<div class="dd-diff-line removed">-${escapeHtml(line)}</div>`).join('')}
        ${newLines.map(line => `<div class="dd-diff-line added">+${escapeHtml(line)}</div>`).join('')}
      `,
    };
  }

  function renderVersionsList(versions, opts) {
    opts = opts || {};
    if (!Array.isArray(versions) || versions.length === 0) return emptyState('暂无版本');
    const sorted = versions.slice().sort(compareVersionTime);
    return `
      <div style="max-height:62vh;overflow:auto;padding:.25rem;">
        ${sorted.slice().reverse().map(v => {
          const idx = sorted.findIndex(item => versionId(item) === versionId(v));
          const prev = idx > 0 ? sorted[idx - 1] : null;
          return `
            <div class="dd-version-row">
              <div style="display:flex;justify-content:space-between;gap:.6rem;margin-bottom:.25rem;">
                <span class="mono" style="font-size:.72rem;overflow-wrap:anywhere;">${escapeHtml(versionId(v) || '')}</span>
                <span class="badge ${v.content_changed ? 'badge-primary' : ''}">${v.content_changed ? 'changed' : 'same'}</span>
              </div>
              <div style="font-size:.75rem;color:var(--text-muted);">${formatDateMs(v.processed_time)} · ${escapeHtml(v.source_document || '')}</div>
              ${renderVersionDiff(v, prev)}
              <details class="dd-version-full">
                <summary>完整内容</summary>
                <div class="md-content" style="font-size:.82rem;margin-top:.45rem;line-height:1.55;">${renderMarkdown(versionText(v) || '空内容')}</div>
              </details>
            </div>
          `;
        }).join('')}
      </div>
    `;
  }

  async function fetchVersions(concept, opts) {
    const fid = familyId(concept);
    if (!fid) throw new Error('缺少 family_id');
    const api = opts?.api || window.state?.api;
    const graphId = opts?.graphId || window.state?.currentGraphId || 'default';
    const res = await api.entityVersions(fid, graphId);
    return res.data?.versions || res.data || [];
  }

  async function openVersionsModal(concept, opts) {
    opts = opts || {};
    ensureStyles();
    const fid = familyId(concept);
    if (!fid) {
      showToast?.('缺少 family_id，无法加载版本历史', 'warning');
      return null;
    }
    const modal = showModal({
      title: `版本历史 - ${truncate(title(concept), 48)}`,
      size: 'lg',
      content: `<div style="padding:1rem;">${spinnerHtml()} 加载版本...</div>`,
    });
    try {
      const versions = await fetchVersions(concept, opts);
      modal.overlay.querySelector('.modal-body').innerHTML = renderVersionsList(versions, opts);
    } catch (err) {
      modal.overlay.querySelector('.modal-body').innerHTML = `<div style="color:var(--danger);">加载版本失败: ${escapeHtml(err.message)}</div>`;
    }
    return modal;
  }

  function renderVersionSwitcher(versions, currentVersionId) {
    if (!versions?.length) return '';
    const sorted = versions.slice().sort(compareVersionTime);
    const currentIdx = Math.max(0, sorted.findIndex(v => v.version_id === currentVersionId));
    return `
      <div class="dd-version-switcher">
        <button class="btn btn-secondary btn-sm" data-version-nav="-1" ${currentIdx <= 0 ? 'disabled' : ''}>上一版</button>
        <select class="input" data-version-select>
          ${sorted.map((v, idx) => `<option value="${escapeAttr(v.version_id)}" ${idx === currentIdx ? 'selected' : ''}>v${v.version_seq || idx + 1} · ${escapeHtml(formatDateMs(v.processed_time))}</option>`).join('')}
        </select>
        <button class="btn btn-secondary btn-sm" data-version-nav="1" ${currentIdx >= sorted.length - 1 ? 'disabled' : ''}>下一版</button>
      </div>
    `;
  }

  async function loadInlineVersionSwitcher(root, concept, opts) {
    opts = opts || {};
    ensureStyles();
    const box = root.querySelector('[data-concept-version-switcher]');
    if (!box || !familyId(concept)) return;
    try {
      const versions = await fetchVersions(concept, opts);
      const renderAndBind = (selectedId) => {
        const selected = versions.find(v => v.version_id === selectedId) || concept;
        box.innerHTML = renderVersionSwitcher(versions, selected.version_id || versionId(selected));
        const content = root.querySelector('[data-concept-version-content]');
        if (content) content.innerHTML = renderMarkdown(versionText(selected) || '暂无内容');
        const evidenceBox = root.querySelector('[data-concept-version-evidence]');
        if (evidenceBox && opts.evidenceForVersion) {
          const evidenceHtml = renderEvidenceList(opts.evidenceForVersion(selected.version_id));
          evidenceBox.style.display = evidenceHtml ? '' : 'none';
          evidenceBox.innerHTML = evidenceHtml ? `<div style="font-weight:600;margin-bottom:.35rem;">原文句子</div>${evidenceHtml}` : '';
        }
        bindSwitcher(box, versions, renderAndBind);
      };
      renderAndBind(concept.version_id || versionId(concept));
    } catch (err) {
      box.innerHTML = `<span style="color:var(--danger);">版本加载失败: ${escapeHtml(err.message)}</span>`;
    }
  }

  function bindSwitcher(box, versions, apply) {
    const select = box.querySelector('[data-version-select]');
    if (select) select.addEventListener('change', () => apply(select.value));
    box.querySelectorAll('[data-version-nav]').forEach(btn => {
      btn.addEventListener('click', () => {
        const current = box.querySelector('[data-version-select]');
        const idx = versions.findIndex(v => v.version_id === current?.value);
        const next = versions[Math.max(0, Math.min(versions.length - 1, idx + Number(btn.getAttribute('data-version-nav') || 0)))];
        if (next) apply(next.version_id);
      });
    });
  }

  function bindPanel(root, concept, opts) {
    opts = opts || {};
    ensureStyles();
    if (opts.inlineVersionSwitcher) loadInlineVersionSwitcher(root, concept, opts);
    if (concept.role === 'relation') hydrateRelationEndpoints(root, concept, opts).catch(() => {});
    root.querySelector('[data-concept-action="versions"]')?.addEventListener('click', () => openVersionsModal(concept, opts));
    root.querySelector('[data-concept-action="edit"]')?.addEventListener('click', () => openEditModal(concept, opts));
    root.querySelector('[data-concept-action="more"]')?.addEventListener('click', () => openConceptModal(concept, opts));
    root.querySelector('[data-concept-action="focus"]')?.addEventListener('click', () => opts.onFocus?.(familyId(concept)));
    root.querySelector('[data-concept-action="exit-focus"]')?.addEventListener('click', () => opts.onExitFocus?.());
    if (window.lucide) lucide.createIcons({ nodes: [root] });
  }

  function openEditModal(concept, opts) {
    opts = opts || {};
    ensureStyles();
    const modal = showModal({
      title: `编辑 - ${truncate(title(concept), 48)}`,
      size: 'lg',
      content: `
        <div style="display:flex;flex-direction:column;gap:.75rem;">
          <label class="form-label">名称</label>
          <input class="input" id="concept-edit-name" value="${escapeAttr(concept.name || '')}">
          <label class="form-label">内容</label>
          <textarea class="input" id="concept-edit-content" style="min-height:220px;">${escapeHtml(versionText(concept) || '')}</textarea>
          <div style="color:var(--text-muted);font-size:.8rem;">保存会追加一个新版本，不会覆盖旧版本。</div>
        </div>
      `,
      footer: `
        <button class="btn btn-secondary btn-sm concept-edit-cancel">取消</button>
        <button class="btn btn-primary btn-sm concept-edit-save">保存为新版本</button>
      `,
    });
    modal.overlay.querySelector('.concept-edit-cancel')?.addEventListener('click', () => modal.close());
    modal.overlay.querySelector('.concept-edit-save')?.addEventListener('click', async () => {
      const api = opts.api || window.state?.api;
      const graphId = opts.graphId || window.state?.currentGraphId || 'default';
      const fid = familyId(concept);
      try {
        const payload = {
          name: modal.overlay.querySelector('#concept-edit-name')?.value || '',
          content: modal.overlay.querySelector('#concept-edit-content')?.value || '',
        };
        await api.updateEntity(fid, payload, graphId);
        showToast?.('概念已保存为新版本', 'success');
        modal.close();
        opts.onEdited?.(fid);
      } catch (err) {
        showToast?.('保存概念失败: ' + err.message, 'error');
      }
    });
    if (window.lucide) lucide.createIcons({ nodes: [modal.overlay] });
    return modal;
  }

  function openConceptModal(concept, opts) {
    opts = opts || {};
    ensureStyles();
    const modal = showModal({
      title: title(concept),
      size: 'lg',
      content: renderConceptBody(concept, {
        ...opts,
        inlineVersionSwitcher: false,
        moreButton: false,
        focusButton: false,
        exitFocusButton: false,
        extraHtml: renderProvenance(opts.provenance),
      }),
    });
    bindPanel(modal.overlay, concept, opts);
    return modal;
  }

  function renderProvenance(provenance) {
    if (!Array.isArray(provenance) || !provenance.length) return '';
    return `
      <div class="dd-concept-section">
        <div style="font-weight:600;margin-bottom:.4rem;">Provenance edges</div>
        ${provenance.slice(0, 30).map(e => pill(e.edge_type || 'edge')).join('') || emptyState('暂无溯源')}
      </div>
    `;
  }

  function normalizeConceptInput(input, role, opts) {
    if (!input || typeof input !== 'string') return input;
    const candidates = opts?.candidates || [];
    return candidates.find(item => versionId(item) === input || familyId(item) === input) || { id: input, version_id: input, family_id: input, role };
  }

  return {
    ensureStyles,
    versionId,
    familyId,
    title,
    endpointInfo,
    renderConceptBody,
    renderEvidenceList,
    renderVersionsList,
    renderVersionDiff,
    loadInlineVersionSwitcher,
    bindPanel,
    openConceptModal,
    openVersionsModal,
    openEditModal,
    normalizeConceptInput,
  };
})();
