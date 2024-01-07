<!-- A Svelte component that takes in a JSON file of vectors and visualizes them as a 2D scatterplot.--> 
<script lang='ts'>
	import { onMount } from 'svelte';
	import { scaleLinear } from 'd3-scale';
    import * as d3 from 'd3';
	import { UMAP } from 'umap-js';
    import { interpolateBlues } from 'd3-scale-chromatic';

	export let embeddings = [] as number[][];
    export let selectedIDs = [] as number[];


	let svg: SVGElement;
	let width = 200;
	let height = 100;
	let interpretable = true;
	let xLabel = 'Level 0 Entropy';
	let yLabel = 'Level 1 Entropy';

	const padding = { top: 20, right: 40, bottom: 40, left: 40 };

	// UMAP the embeddings to 2D if they are not already
    $: if (embeddings.length > 0 && embeddings[0].length > 2) {
        const umap = new UMAP();
        embeddings = umap.fit(embeddings);
        interpretable = false;
        xLabel = 'UMAP Dimension 1';
        yLabel = 'UMAP Dimension 2';
	}


    // get min of the 0th index of each item in embeddings
    $: yMin = Math.min(...embeddings.map(item => item[1]))
    $: yMax = Math.max(...embeddings.map(item => item[1]))
    $: xMin = Math.min(...embeddings.map(item => item[0]))
    $: xMax = Math.max(...embeddings.map(item => item[0]))

	$: xScale = scaleLinear()
		.domain([xMin, xMax])
		.range([padding.left, width - padding.right]);

	$: yScale = scaleLinear()
		.domain([yMin, yMax])
		.range([height - padding.bottom, padding.top]);

    $: xTicks = Array.from({length: xMax - xMin + 1}, (_, i) => i + xMin)
    $: yTicks = Array.from({length: yMax - yMin + 1}, (_, i) => i + yMin)


	onMount(resize);

	function resize() {
		({ width, height } = svg.getBoundingClientRect());
	}

    function pointColor(index: number) {
        return interpolateBlues(embeddings[index].reduce((sum, value) => sum + value, 0) / embeddings[index].length);
    }
</script>

<svelte:window on:resize={resize} />

<svg bind:this={svg}>
	{#if interpretable}
		<!-- y axis -->
		<g class="axis y-axis">
				{#each yTicks as tick}
					<g class="tick tick-{tick}" transform="translate(0, {yScale(tick)})">
						<line x1={padding.left} x2={xScale(22)} />
						<text x={padding.left - 8} y="+4">{tick}</text>
					</g>
				{/each}
		</g>

		<!-- x axis -->
		<g class="axis x-axis">
			{#each xTicks as tick}
				<g class="tick" transform="translate({xScale(tick)},0)">
					<line y1={yScale(0)} y2={yScale(13)} />
					<text y={height - padding.bottom + 16}>{tick}</text>
				</g>
			{/each}
		</g>
	{/if}

	<!-- data -->
	{#each embeddings as embedding, index}
        <circle
            id={index.toString()}
            cx={xScale(embedding[0])} 
            cy={yScale(embedding[1])} 
            r="4"
            role="none"
            class={selectedIDs.includes(index) ? 'selected' : ''}
            fill={pointColor(index)}
            stroke={pointColor(index)}
            on:mouseover="{event => {
                d3.select(event.target).attr('r', 8);
            //    instanceID = event.target.id;
            }}"
            on:focus="{event => {
                d3.select(event.target).attr('r', 8);
            //    instanceID = event.target.id;
            }}"
            on:mouseout="{event => d3.select(event.target).attr('r', 4)}"
            on:blur="{event => d3.select(event.target).attr('r', 4)}"
            on:click="{event => {
                if (selectedIDs.includes(parseInt(event.target.id))) {
                    selectedIDs = selectedIDs.filter(id => id !== parseInt(event.target.id));
                } else {
                    selectedIDs = [...selectedIDs, parseInt(event.target.id)]
                }
            }}"
        />
	{/each}

    <!--- axis labels -->
    <text x={width / 2} y={height - 10} text-anchor="middle">{xLabel}</text>
    <text x='10' y={(height / 2) - 10} transform="rotate(-90, 10, {(height / 2) - 10})" text-anchor="middle">{yLabel}</text>
</svg>

<style>
	svg {
		width: 100%;
		height: 25vh;
		float: left;
	}

	circle {
		fill-opacity: 0.6;
	}

	.tick line {
		stroke: #ddd;
		stroke-dasharray: 2;
	}

	text {
		font-size: 12px;
		fill: #999;
	}

	.x-axis text {
		text-anchor: middle;
	}

	.y-axis text {
		text-anchor: end;
	}

    .selected {
        fill: red;
    }
</style>
