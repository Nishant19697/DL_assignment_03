import wandb
import numpy as np
from jinja2 import Template

def make_q6(data):
    # Prepare all samples
    all_samples = [
        {
            "source_tokens": list(sample["source_word"]),
            "target_tokens": list(sample["target_word"]),
            "attention_matrix": sample["attention_map"].tolist()
        }
        for sample in data
    ]

    config = {
        "samples": all_samples
    }
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>Token Attention Map</title>
    <style>
        body {
            background: #1a1a2e;
            font-family: Arial, sans-serif;
            color: #e0e0e0;
            padding: 30px;
        }
        h1 {
            text-align: center;
            color: #fca311;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .sample-button {
            background: #14213d;
            color: #ffffff;
            border: 1px solid #fca311;
            border-radius: 5px;
            padding: 10px 15px;
            margin: 5px;
            cursor: pointer;
            transition: background 0.2s ease;
        }
        .sample-button:hover {
            background: #fca311;
            color: #000;
        }
        .visual-container {
            display: flex;
            justify-content: space-between;
            position: relative;
            margin-top: 40px;
        }
        .token-column {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }
        .token-box {
            background: #264653;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid transparent;
        }
        .token-box:hover {
            border-color: #f4a261;
        }
        .source-token {
            background-color: #2a9d8f;
        }
        .target-token {
            background-color: #e76f51;
        }
        svg {
            position: absolute;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }
        .attention-line {
            stroke: #fca311;
            stroke-width: 2;
            fill: none;
            opacity: 0.5;
        }
    </style>
    </head>
    <body>

    <h1>Token-Level Attention Map</h1>
    <div class="controls" id="controls"></div>

    <div class="visual-container" id="visual-wrapper">
        <div class="token-column" id="source-column"></div>
        <svg id="attention-svg"></svg>
        <div class="token-column" id="target-column"></div>
    </div>

    <script>
    const samples = {{ samples | tojson }};
    let currentSample = samples[0];

    const controls = document.getElementById("controls");
    const sourceCol = document.getElementById("source-column");
    const targetCol = document.getElementById("target-column");
    const svg = document.getElementById("attention-svg");
    const wrapper = document.getElementById("visual-wrapper");

    samples.forEach((sample, idx) => {
        const btn = document.createElement("button");
        btn.className = "sample-button";
        btn.textContent = sample.source_tokens.join('') + " → " + sample.target_tokens.join('');
        btn.onclick = () => {
            currentSample = sample;
            renderView();
        };
        controls.appendChild(btn);
    });

    function renderView() {
        sourceCol.innerHTML = '';
        targetCol.innerHTML = '';
        svg.innerHTML = '';

        currentSample.source_tokens.forEach((token, i) => {
            const el = document.createElement('div');
            el.className = 'token-box source-token';
            el.textContent = token;
            el.dataset.index = i;
            el.onmouseenter = () => drawAttention(i);
            el.onmouseleave = clearAttention;
            sourceCol.appendChild(el);
        });

        currentSample.target_tokens.forEach((token) => {
            const el = document.createElement('div');
            el.className = 'token-box target-token';
            el.textContent = token;
            targetCol.appendChild(el);
        });
    }

    function drawAttention(sourceIndex) {
        clearAttention();
        const svgNS = "http://www.w3.org/2000/svg";
        const sourceEls = sourceCol.querySelectorAll('.token-box');
        const targetEls = targetCol.querySelectorAll('.token-box');

        const wrapperRect = wrapper.getBoundingClientRect();

        const src = sourceEls[sourceIndex].getBoundingClientRect();
        const srcX = src.right - wrapperRect.left;
        const srcY = src.top + src.height / 2 - wrapperRect.top;

        currentSample.attention_matrix[sourceIndex].forEach((weight, targetIndex) => {
            if (weight > 0.1) {
                const tgt = targetEls[targetIndex].getBoundingClientRect();
                const tgtX = tgt.left - wrapperRect.left;
                const tgtY = tgt.top + tgt.height / 2 - wrapperRect.top;

                const path = document.createElementNS(svgNS, "path");
                const dx = (tgtX - srcX) / 2;
                const d = `M${srcX},${srcY} C${srcX + dx},${srcY} ${tgtX - dx},${tgtY} ${tgtX},${tgtY}`;
                path.setAttribute("d", d);
                path.setAttribute("stroke-opacity", weight);
                path.classList.add("attention-line");
                svg.appendChild(path);
            }
        });
    }

    function clearAttention() {
        svg.innerHTML = '';
    }

    renderView();
    </script>

    </body>
    </html>

    """


    html_content = Template(HTML_TEMPLATE).render(**config)

    wandb.log({
        "attention_visualization_02": wandb.Html(html_content, inject=True)
    })

    print("Logged interactive attention visualization with all words.")

