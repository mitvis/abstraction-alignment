<!-- A Svelte component that takes in a JSON file of vectors and visualizes them as a 2D scatterplot.--> 
<script lang='ts'>
	import { onMount, afterUpdate } from 'svelte';
	import { scaleLinear } from 'd3-scale';
    import * as d3 from 'd3';
	import { UMAP } from 'umap-js';
    import { interpolateBlues } from 'd3-scale-chromatic';
	import type { EmbeddingPoint } from './types';
	import type { VisualizationSpec } from "svelte-vega";
  	import { Vega } from "svelte-vega";
	import vegaEmbed from 'vega-embed';
	import * as vega from 'vega';

	export let embeddings = [] as number[][];
    export let selectedIDs = [] as number[];

	let view: vega.View;
	const width = 425;
	const height = 200;
	const padding = 10;
	let interpretable = true;
	let xLabel = 'Level 0 Entropy';
	let yLabel = 'Level 1 Entropy';
	let scaledEmbeddings = [] as number[][];

	let data = {table: [] as EmbeddingPoint[]};
	let spec: VisualizationSpec

	// UMAP the embeddings to 2D if they are not already
    $: if (embeddings.length > 0 && embeddings[0].length > 2) {
        const umap = new UMAP();
        scaledEmbeddings = umap.fit(embeddings);
        interpretable = false;
        xLabel = 'UMAP Dimension 1';
        yLabel = 'UMAP Dimension 2';
	} else {
		scaledEmbeddings = embeddings;
	}

	$: if (scaledEmbeddings.length > 0) {
		data.table = scaledEmbeddings.map((scaledEmbedding, index) => {
			return {
				'x': scaledEmbedding[0],
				'y': scaledEmbedding[1],
				'id': index,
				'selected': selectedIDs.includes(index),
				'Mean Entropy': embeddings[index].reduce((sum, value) => sum + value, 0) / embeddings[index].length
			}
		});
	}

	// $: spec = {
	// 	$schema: 'https://vega.github.io/schema/vega-lite/v5.json',
	// 	data: {
	// 		name: 'table'
	// 	},
	// 	mark: 'point',
	// 	encoding: {
	// 		x: { field: 'x', type: 'quantitative', title: xLabel},
	// 		y: { field: 'y', type: 'quantitative', title: yLabel},
	// 		tooltip: { field: 'id', type: 'nominal' },
	// 		color: { field: 'Mean Entropy', type: 'quantitative' },
	// 		opacity: {
	// 			condition: { test: 'datum.selected', value: 1 },
	// 			value: 0.05
	// 		},
	// 		zindex: { 
	// 			condition: { test: 'datum.selected', value: 1 },
	// 			value: 0,
	// 		}
	// 	},
	// 	height: height,
	// 	width: width,
	// 	selection: {
	// 		selectedPoint: { type: 'single', fields: ['id'] }
	// 	}
	// };

	$: spec = {
		$schema: 'https://vega.github.io/schema/vega/v5.json',
		height: height,
		width: width,
		padding: padding,
		config: {
			axis: {
				domain: false,
			}
		},
		signals: [
			{
				name: "hover",
				on: [
					{events: "*:pointerover", encode: "hover"},
					{events: "*:pointerout", encode: "leave"},
					{events: "*:pointerdown", encode: "select"},
					{events: "*:pointerup", encode: "release"}
				]
			},
			{ name: "xrange", update: "[0, width]" },
    		{ name: "yrange", update: "[height, 0]" },
			{
				name: "down",
				value: null,
				on: [
					{events: "touchend", update: "null"},
					{events: "pointerdown, touchstart", update: "xy()"}
				]
			},
			{
				name: "xcur",
				value: null,
				on: [
					{
						events: "pointerdown, touchstart, touchend", 
						update: "slice(xdom)"
					}
				]
			},
			{
				name: "ycur",
				value: null,
				on: [
					{
						events: "pointerdown, touchstart, touchend", 
						update: "slice(ydom)"
					}
				]
			},
			{
				name: "delta",
				value: [0, 0],
				on: [
					{
						events: [
							{
								source: "window", type: "pointermove", consume: true,
								between: [{type: "pointerdown"}, {source: "window", type: "pointerup"}]
							},
							{
								type: "touchmove", consume: true,
								filter: "event.touches.length === 1",
							}
						],
						update: "down ? [down[0]-x(), down[1]-y()] : [0,0]"
					}
				]
			},
			{
				name: "anchor",
				value: [0, 0],
				on: [
					{
						events: "wheel",
						update: "[invert('xscale', x()), invert('yscale', y())]"
					},
					{
						events: {type: "touchstart", filter: "event.touches.length===2"},
						update: "[(xdom[0] + xdom[1]) / 2, (ydom[0] + ydom[1]) / 2]"
					}
				]
			},
			{
				name: "zoom",
				value: 1,
				on: [
					{
						events: "wheel!",
						force: true,
						update: "pow(1.001, event.deltaY * pow(16, event.deltaMode))"
					},
					{
						events: {"signal": "dist2"},
						force: true,
						update: "dist1 / dist2"
					}
				]
			},
			{
				name: "dist1",
				value: 0,
				on: [
					{
						events: {type: "touchstart", filter: "event.touches.length === 2"},
						update: "pinchDistance(event)"
					},
					{
						events: {signal: "dist2"},
						update: "dist2"
					}
				]
			},
			{
				name: "dist2",
				value: 0,
				on: [{
					events: {type: "touchmove", consume: true, filter: "event.touches.length === 2"},
					update: "pinchDistance(event)"
				}
				]
			},
			{
				name: "xdom",
				update: "slice(xext)",
				on: [
					{
						events: {signal: "delta"},
						update: "[xcur[0] + span(xcur) * delta[0] / width, xcur[1] + span(xcur) * delta[0] / width]"
					},
					{
						events: {signal: "zoom"},
						update: "[anchor[0] + (xdom[0] - anchor[0]) * zoom, anchor[0] + (xdom[1] - anchor[0]) * zoom]"
					}
				]
			},
			{
				name: "ydom",
				update: "slice(yext)",
				on: [
					{
						events: {signal: "delta"},
						update: "[ycur[0] - span(ycur) * delta[1] / height, ycur[1] - span(ycur) * delta[1] / height]"
					},
					{
						events: {signal: "zoom"},
						update: "[anchor[1] + (ydom[0] - anchor[1]) * zoom, anchor[1] + (ydom[1] - anchor[1]) * zoom]"
					}
				]
			},
			{
				name: "size",
				update: "clamp(20 / span(xdom), 0, 1000)"
			}
		],
		data: [
			{
				name: 'table',
				transform: [
					{type: "extent", field: "x", signal: "xext"},
					{type: "extent", field: "y", signal: "yext"},
				]
			}
		],
		scales: [
			{
				name: "xscale",
				zero: false,
				domain: {signal: "xdom"},
				range: {signal: "xrange"}
			},
			{
				name: "yscale",
				zero: false,
				domain: {signal: "ydom"},
				range: {signal: "yrange"}
			},
		],
		axes: [
			{
				scale: "xscale",
				orient: "bottom",
				grid: true,
				title: xLabel,
			},
			{
				scale: "yscale",
				orient: "left",
				grid: true,
				title: yLabel,
			}
		],
		marks: [
			{
				name: "marks",
				type: 'symbol',
				from: { data: 'table' },
				clip: true,
				encode: {
					update: {
						x: { scale: 'xscale', field: 'x' },
						y: { scale: 'yscale', field: 'y' },
						size: {value: 5},
						shape: {value: "circle"},
						strokeWidth: {value: 2},
						opacity: {value: 0.5},
						stroke: {value: "#4682b4"},
						fill: {value: "transparent"}
					},
				},
			}
		]
	}
		

	// onMount(async () => {
	// 	console.log('onMount')
	// 	const result = await vegaEmbed('#projection-plot', spec);
	// 	view = result.view;

	// 	// view.addSignalListener('selectedPoint', function(_, value) {
    //     //     if (value) {
	// 	// 		if (selectedIDs.includes(value.id[0])) {
	// 	// 			selectedIDs = selectedIDs.filter(id => id != value.id[0]);
	// 	// 		} else {
	// 	// 			selectedIDs = [...selectedIDs, value.id[0]];
	// 	// 		}
    //     //     }
    //     // });
	// })

	// $: if (view) {
    //     view.change('table', vega.changeset().remove(() => true).insert(data['table'])).run();
    // }

    function pointColor(index: number) {
        return interpolateBlues(embeddings[index].reduce((sum, value) => sum + value, 0) / embeddings[index].length);
    }
</script>

<div id='projection-plot'></div>
<Vega data={data} spec={spec} />

