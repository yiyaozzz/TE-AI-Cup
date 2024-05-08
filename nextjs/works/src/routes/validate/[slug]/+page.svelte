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
<body>
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
</body>

<style>
	body {
		font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
		background-color: #e8effc;
		margin: 0;
		padding: 20px;
		color: #333;
	}

	h1 {
		color: #023047;
		text-align: center;
	}

	img {
		max-width: calc(50% - 20px);
		border-radius: 8px;
		transition: transform 0.3s ease;
	}

	img:hover {
		transform: scale(1.03);
	}

	textarea {
		width: 100%;
		height: 120px;
		padding: 12px;
		border: none;
		border-radius: 8px;
		background-color: #f1f5f9;
		box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
		resize: none; /* Disable resizing */
	}

	form {
		background-color: #fff;
		padding: 20px;
		border-radius: 8px;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
		margin-top: 20px;
	}

	input[type='text'],
	button {
		width: 100%;
		padding: 12px;
		margin: 10px 0;
		border: 1px solid #ccc;
		border-radius: 8px;
		transition: all 0.3s ease;
	}

	input[type='text']:focus {
		outline: none;
		border-color: #0366d6;
		box-shadow: 0 0 0 3px rgba(3, 102, 214, 0.3);
	}

	button {
		background-color: #023e8a;
		color: white;
		border: none;
		cursor: pointer;
		transition: background-color 0.3s ease;
	}

	button:hover {
		background-color: #03045e;
	}

	div {
		margin-bottom: 16px;
	}

	label {
		font-weight: bold;
		margin-bottom: 5px;
		color: #023047;
	}
</style>
