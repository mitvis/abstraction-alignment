<!-- Checks whether a component intersects with the viewport.
     From: https://github.com/donovanh/svelte-image-loading  -->
<script>
	import { onMount } from 'svelte';
	export let once = false;
	export let top = 0;
	export let bottom = 0;
	export let left = 0;
	export let right = 0;
    let intersecting = false;
  
	/**
     * @type {Element}
     */
	let container;
	onMount(() => {
		if (typeof IntersectionObserver !== 'undefined') {
			const rootMargin = `${bottom}px ${left}px ${top}px ${right}px`;
			const observer = new IntersectionObserver(entries => {
				intersecting = entries[0].isIntersecting;
				if (intersecting && once) {
					observer.unobserve(container);
				}
			}, {
				rootMargin
			});
			observer.observe(container);
			return () => observer.unobserve(container);
		}
	});
</script>

<style>
	div {
		width: 100%;
		height: 100%;
	}
</style>

<div bind:this={container}>
	<slot {intersecting}></slot>
</div>