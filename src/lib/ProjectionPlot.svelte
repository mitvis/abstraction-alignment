<!-- A Svelte component that takes in a JSON file of vectors and visualizes them as a 2D scatterplot.--> 
<script lang='ts'>
	import { onMount } from 'svelte';
	import { UMAP } from 'umap-js';
	import type { EmbeddingPoint } from './types';
	import type { VisualizationSpec } from "svelte-vega";
	import vegaEmbed from 'vega-embed';
	import * as vega from 'vega';

	export let embeddings = [] as number[][];
    export let selectedIDs: number[];

	let view: vega.View;
	const width = 475;
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
		data.table = scaledEmbeddings.map((scaledEmbedding, i) => {
			return {
				'x': scaledEmbedding[0],
				'y': scaledEmbedding[1],
				'id': i,
				'selected': selectedIDs.includes(i),
				'drop-off': (embeddings[i][0] - embeddings[i][embeddings[i].length - 1]) / embeddings[i][0], 
			}
		});
	}

	$: spec = {
		$schema: 'https://vega.github.io/schema/vega-lite/v5.json',
		data: {
			name: 'table'
		},
		mark: 'circle',
		encoding: {
			x: { field: 'x', type: 'quantitative', title: xLabel},
			y: { field: 'y', type: 'quantitative', title: yLabel},
			tooltip: { field: 'id', type: 'nominal' },
			color: {field: "drop-off", type: "quantitative"},
			opacity: {
				condition: { test: 'datum.selected', value: 1 },
				value: 0.05
			},
			zindex: { 
				condition: { test: 'datum.selected', value: 1 },
				value: 0,
			}
		},
		height: height,
		width: width,
		selection: {
			selectedPoint: { 
				type: 'single', 
				fields: ['id'],
				on: "mousedown[!event.metaKey]"
			},
			brush: {
				type: "interval",
				resolve: "union",
				on: "[mousedown[event.shiftKey], window:mouseup] > window:mousemove!",
				translate: "[mousedown[event.shiftKey], window:mouseup] > window:mousemove!",
				zoom: "null"
			},
			panzoom: {
				type: "interval", 
				bind: "scales",
				translate: "[mousedown[event.metaKey], window:mouseup] > window:mousemove!",
				zoom: "wheel!"
			}
		}
	};

	function defaultSelectedIDs() {
		return Array.from({length: data.table.length}, (_, i) => i);
	};
		

	onMount(async () => {
		const result = await vegaEmbed('#projection-plot', spec, {actions: false});
		view = result.view;

		view.addSignalListener('selectedPoint', async function(_, value) {
			if (Object.keys(value).length === 0) {
				selectedIDs = defaultSelectedIDs();
			} else {
				if (value) {
					if (selectedIDs.length == data.table.length) { // everything is selected
						selectedIDs = [value.id[0]];
					} else { // a subset is selected
						if (selectedIDs.includes(value.id[0])) {
							if (selectedIDs.length == 1) { // clicked point is the only thing selected
								selectedIDs = defaultSelectedIDs();
							} else {
								selectedIDs = selectedIDs.filter(id => id != value.id[0]);
							};
						} else {
							selectedIDs = [...selectedIDs, value.id[0]];
						}
					}
				}
			}
        });

		view.addSignalListener('brush', function(_, value) {
			if (Object.keys(value).length === 0) {
				selectedIDs = defaultSelectedIDs();
			} else {
				if (value) {
					// Get the extents of the brush selection
					let [xMin, xMax] = value.x;
					let [yMin, yMax] = value.y;

					// Filter the data points that are within the brush selection
					let selectedPoints = data['table'].filter(d => 
						d.x >= xMin && d.x <= xMax && d.y >= yMin && d.y <= yMax
					);

					// Get the IDs of the selected points
					let selectedPointIDs = selectedPoints.map(d => d.id);

					// Add the selected point IDs to selectedIDs
					if (selectedPointIDs.length == 0) {
						selectedIDs = defaultSelectedIDs();
					} else {
						selectedIDs = [...selectedPointIDs];
					}
				}
			}
		});
	});


	$: if (view) {
        view.change('table', vega.changeset().remove(() => true).insert(data['table'])).run();
		view.resize();
    }

</script>

<div id='projection-plot'></div>
<!-- <Vega data={data} spec={spec} /> -->

