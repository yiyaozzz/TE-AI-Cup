import { promises as fs } from 'fs';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import util from 'node:util';
import { exec } from 'node:child_process';

const execAsync = util.promisify(exec);
// Define the uploads directory relative to the current file
const UPLOAD_DIR = path.resolve('uploads');

// Ensure the upload directory exists
fs.mkdir(UPLOAD_DIR, { recursive: true }).catch(console.error);

export async function POST({ request }) {
	const formData = await request.formData();
	const file = formData.get('file');

	if (file) {
		// We use the original file name here. In a real application, you might want
		// to rename the file to avoid conflicts or security issues.
		const fileName = `${uuidv4()}.pdf`;
		const filePath = path.join(UPLOAD_DIR, fileName);
		console.log('Saving to ' + filePath);

		// Read the file stream and write it to a new file
		const arrayBuffer = await file.arrayBuffer();
		try {
			await fs.writeFile(filePath, Buffer.from(arrayBuffer));

			// Process file here
			const { stdout, stderr } = await execAsync(`python3 process_scripts.py ${filePath}`);
			console.log('stdout:', stdout);
			console.error('stderr:', stderr);
			return new Response(
				JSON.stringify({ message: 'File uploaded successfully', fileId: fileName }),
				{
					status: 200,
					headers: {
						'Content-Type': 'application/json'
					}
				}
			);
		} catch (error) {
			console.error('Error writing file:', error);
			return new Response(JSON.stringify({ error: 'Failed to save file' }), {
				status: 500,
				headers: {
					'Content-Type': 'application/json'
				}
			});
		}
	}

	return new Response(JSON.stringify({ error: 'No file uploaded' }), {
		status: 400,
		headers: {
			'Content-Type': 'application/json'
		}
	});
}
