<script>
    export let dimensions = 0;
    $: queryVector = Array.from({length: dimensions}, () => 0);
</script>

<div id='query'>
    <!-- create an input value for each {#each queryVector} -->
    <p>Query vector:</p>
    <div>
        {#each queryVector as value, i}
            <input class='query-dimension' bind:value={queryVector[i]} />
        {/each}
    </div>
</div>

<style>
 #query {
    display: flex;
    flex-direction: row;
    column-gap: 1em;
    align-items: center;
 }
 .query-dimension {
    width: 2em;
 }
</style>