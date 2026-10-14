// ApexInfer Interactive UI Controller

const blockGrid = document.getElementById("blockGrid");
const activeBlocksBadge = document.getElementById("activeBlocksBadge");
const promptInput = document.getElementById("promptInput");
const maxTokensInput = document.getElementById("maxTokensInput");
const generateBtn = document.getElementById("generateBtn");
const outputContainer = document.getElementById("outputContainer");
const statsFooter = document.getElementById("statsFooter");
const statThroughput = document.getElementById("statThroughput");
const statAcceptance = document.getElementById("statAcceptance");
const statPasses = document.getElementById("statPasses");
const metricSpeedup = document.getElementById("metricSpeedup");
const metricReduction = document.getElementById("metricReduction");

const TOTAL_BLOCKS = 64;
let blocks = [];

// Initialize 64 Physical Blocks in the Memory Map
function initBlockGrid() {
    blockGrid.innerHTML = "";
    blocks = [];
    for (let i = 0; i < TOTAL_BLOCKS; i++) {
        const cell = document.createElement("div");
        cell.className = "block-cell w-full h-full rounded bg-slate-800/80 border border-slate-700/40 hover:border-cyan-400/60 cursor-pointer";
        cell.title = `Physical Block #${i}: Free`;
        blockGrid.appendChild(cell);
        blocks.push(cell);
    }
}
initBlockGrid();

function updateMemoryMap(allocatedCount, activeCount) {
    activeBlocksBadge.innerText = `${allocatedCount} / ${TOTAL_BLOCKS} Blocks`;
    for (let i = 0; i < TOTAL_BLOCKS; i++) {
        const cell = blocks[i];
        if (i < activeCount) {
            cell.className = "block-cell w-full h-full rounded bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)] border border-indigo-400";
            cell.title = `Physical Block #${i}: Active KV Sequence`;
        } else if (i < allocatedCount) {
            cell.className = "block-cell w-full h-full rounded bg-cyan-500/80 border border-cyan-400/60";
            cell.title = `Physical Block #${i}: Allocated (Reserved)`;
        } else {
            cell.className = "block-cell w-full h-full rounded bg-slate-800/80 border border-slate-700/40";
            cell.title = `Physical Block #${i}: Free`;
        }
    }
}

generateBtn.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    const maxTokens = parseInt(maxTokensInput.value) || 35;
    if (!prompt) return;

    generateBtn.disabled = true;
    generateBtn.classList.add("opacity-50", "cursor-not-allowed");
    outputContainer.innerHTML = `<span class="text-cyan-400 animate-pulse font-mono">⚡ Running speculative decoding (Draft + Target verification)...</span>`;

    // Animate memory allocation
    let blocksNeeded = Math.ceil((prompt.length / 4 + maxTokens) / 16);
    updateMemoryMap(blocksNeeded + 2, blocksNeeded);

    try {
        const response = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, max_tokens: maxTokens, gamma: 4 })
        });

        const data = await response.json();

        // Render generated text
        outputContainer.innerHTML = `<span class="text-slate-400">Prompt: ${prompt}</span>\n\n<span class="text-slate-100 font-semibold">${data.output_text}</span>`;
        
        // Update stats
        statsFooter.classList.remove("hidden");
        statThroughput.innerText = `${data.tokens_per_sec.toFixed(1)} tok/s`;
        statAcceptance.innerText = `${data.acceptance_rate_pct.toFixed(1)}%`;
        statPasses.innerText = `${data.target_forward_passes} passes (${data.tokens_per_target_pass.toFixed(2)} tok/pass)`;

        metricSpeedup.innerText = `${data.tokens_per_target_pass.toFixed(1)}x`;
        const reduction = Math.max(0, (1 - data.target_forward_passes / data.tokens_generated) * 100);
        metricReduction.innerText = `${reduction.toFixed(0)}%`;

        // Pulse memory map to show deallocation upon sequence completion
        setTimeout(() => {
            updateMemoryMap(0, 0);
        }, 2000);

    } catch (e) {
        outputContainer.innerHTML = `<span class="text-rose-400 font-mono">Error running inference: ${e.message}</span>`;
    } finally {
        generateBtn.disabled = false;
        generateBtn.classList.remove("opacity-50", "cursor-not-allowed");
    }
});
