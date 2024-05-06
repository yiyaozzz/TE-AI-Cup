import { error, redirect } from '@sveltejs/kit';
import util from 'node:util';
import { exec } from 'node:child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { goto } from '$app/navigation';

const execAsync = util.promisify(exec);
// Define the uploads directory relative to the current file
const PROCESS_DIR = path.resolve('processing');

/** @type {import('./$types').PageServerLoad} */
export async function load({ params }) {
	if (!checkIfFileExists(`${params.slug}.json`)) {
		error(404, 'File not found');
		return;
	}

	const { stdout, stderr } = await execAsync(`python3 validate.py processing/${params.slug}.json`);
	console.log('stdout:', stdout);
	console.error('stderr:', stderr);

	const jsonData = await fs.readFile(`processing/${params.slug}.json`, 'utf8');

	const value = JSON.parse(jsonData);

	console.log('Validate.py stdout:', stdout, stdout.length);

	if (stderr) {
		error(500, 'File validation error');
	}

	if (stdout.length == 0) {
		// Add the exec commands here
		redirect(302, '/');
	}

	/* `let lines = stdout` is assigning the value of the `stdout` variable to the `lines` variable. This
	allows you to work with the output of the command executed in the `stdout` variable in the
	subsequent code. */
	let lines = stdout.split('\n');

	let str = lines[0].split('/');
	// let str2 = str[2].split('[');

	let folderId = params.slug.slice(0,-4);

	const data = {
		folderId,
		page: str[0],
		row: str[1],
		col: str[2],
		prevPage: lines[1],
		prevURL: `/api/image?folderId=${folderId}&page=${lines[1]}`,
		curURL: `/api/image?folderId=${folderId}&page=${str[0]}`,
		value: value
	};

	return data;

	// const post = await getPostFromDatabase(params.slug);
	// if (post) {
	// 	return post;
	// }
	// error(404, 'File validation error');
}

/**
 * @param {string} fileName
 */
function checkIfFileExists(fileName) {
	// Check if the file exists in uploads folder
	console.log(path.join(PROCESS_DIR, fileName));
	return fs
		.stat(path.join(PROCESS_DIR, fileName))
		.then(() => true)
		.catch(() => false);
}
