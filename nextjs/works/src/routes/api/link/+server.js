import { promises as fs } from 'fs';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import util from 'node:util';
import { fileURLToPath } from 'url';
import { exec } from 'node:child_process';

const execAsync = util.promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const UPLOAD_DIR = path.resolve(__dirname, 'uploads');

// Ensure the upload directory exists
fs.mkdir(UPLOAD_DIR, { recursive: true }).catch(console.error);

export async function POST({ request }) {
    const formData = await request.formData();
    const file = formData.get('file');

    if (file) {
        const fileId = uuidv4(); // A unique file ID
        const inputFileName = `${fileId}.pdf`;
        const outputFileName = `${fileId}.xlsx`;
        const inputFilePath = path.join(UPLOAD_DIR, inputFileName);
        const outputFilePath = path.join(UPLOAD_DIR, outputFileName);

        console.log('Saving to ' + inputFilePath);

        // Read the file stream and write it to a new file
        const arrayBuffer = await file.arrayBuffer();
        try {
            await fs.writeFile(inputFilePath, Buffer.from(arrayBuffer));
            const { stdout, stderr } = await execAsync(`python3 process_scripts.py ${filePath}`);

            // TODO: Convert the PDF file to Excel here.
            // For example, you might call a function like: convertPDFToExcel(inputFilePath, outputFilePath);

            // Generate a public URL for downloading the Excel file
            const downloadUrl = `/downloads/${outputFileName}`; // Adjust the URL path as needed for your routing

            return new Response(
                JSON.stringify({ message: 'File uploaded and converted successfully', downloadUrl: downloadUrl }),
                {
                    status: 200,
                    headers: {
                        'Content-Type': 'application/json'
                    }
                }
            );
        } catch (error) {
            console.error('Error processing file:', error);
            return new Response(JSON.stringify({ error: 'Failed to process file' }), {
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
