<script>
	import { goto, invalidateAll } from '$app/navigation';

	/** @type {import('./$types').PageData} */
	export let data;
	console.log(data);
	let textInput = '';
	async function handleSubmit() {
		const response = await fetch('/api/data', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify(data)
		});

		if (response.ok) {
			console.log('Data sent successfully');
			textInput = ''; // Clear the input after sending
			invalidateAll();
			// location.reload();
			// goto(`/validate/${data.folderId}`); // Redirect to the next page
		} else {
			console.error('Failed to send data');
		}
	}
	function checkListContents(array) {
		let containsStrings = false;
		let containsHashmaps = false;

		array.forEach((item) => {
			if (typeof item === 'string') {
				containsStrings = true;
			} else if (typeof item === 'object' && item !== null && !Array.isArray(item)) {
				containsHashmaps = true;
			}
		});

		return { containsStrings, containsHashmaps };
	}
</script>

<h1>Validate</h1>
<img src={data.prevURL} alt="Reload page!" width="900px" />
<img src={data.curURL} alt="Reload page!" width="900px" />
<p>
	{`Error at page: ${data.page}, row: ${data.row}, col: ${data.col}. Please fix the _flags and submit`}
</p>
<!-- <textarea>{JSON.stringify(data.value, null, 4)}</textarea> -->

<form on:submit|preventDefault={handleSubmit}>
	<div>
		<label>OPR.No</label>
		<input
			type="text"
			bind:value={data.value[data.page][data.row]['1'][0]}
			placeholder="Enter some text"
		/>

		<label>Planned WorkCenter Description</label>
		<input
			type="text"
			bind:value={data.value[data.page][data.row]['2'][0]}
			placeholder="Enter some text"
		/>

		<label>Comp Qty.</label>
		<input
			type="text"
			bind:value={data.value[data.page][data.row]['3'][0]}
			placeholder="Enter some text"
		/>
	</div>
	{#if checkListContents(data.value[data.page][data.row]['4']).containsStrings}
		<label>Scrap Qty. & Desc.</label>
		<input
			type="text"
			bind:value={data.value[data.page][data.row]['4'][0]}
			placeholder="Enter some text"
		/>
	{:else}
		<label>Scrap Qty. & Desc.</label>
		{#each data.value[data.page][data.row]['4'] as pair}
			<div>
				<input type="text" bind:value={pair.name} placeholder="Enter some text" />
				<input type="text" bind:value={pair.value} placeholder="Enter some text" />
			</div>
		{/each}
	{/if}
	<!-- <input type="text" bind:value={textInput} placeholder="Enter some text" /> -->
	<button type="submit">Submit</button>
</form>

<style>
	body {
		font-family: Arial, sans-serif;
		background-color: #f4f4f9;
		margin: 40px;
	}

	h1 {
		color: #333;
	}

	img {
		margin: 10px;
		border: 1px solid #ddd;
		border-radius: 4px;
	}

	textarea {
		width: 100%;
		height: 100px;
		padding: 10px;
		border: 1px solid #ccc;
		border-radius: 4px;
		margin-bottom: 20px;
		box-sizing: border-box; /* Added for CSS reset purposes */
	}

	form {
		background-color: white;
		padding: 20px;
		border-radius: 8px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}

	input[type='text'] {
		width: 100%;
		padding: 10px;
		margin: 8px 0;
		display: inline-block;
		border: 1px solid #ccc;
		border-radius: 4px;
		box-sizing: border-box;
	}

	button {
		width: 100%;
		background-color: #4caf50;
		color: white;
		padding: 14px 20px;
		margin: 8px 0;
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}

	button:hover {
		background-color: #45a049;
	}

	div {
		margin-bottom: 10px;
	}

	label {
		font-weight: bold;
		display: block;
	}
</style>
