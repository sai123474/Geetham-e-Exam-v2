// server.js
require('dotenv').config();

const express = require('express');
const { MongoClient, ServerApiVersion, ObjectId } = require('mongodb');
const path = require('path');
const cors = require('cors');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { Pinecone } = require('@pinecone-database/pinecone');
const { pipeline } = require('@xenova/transformers');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const multer = require('multer');
const CryptoJS = require('crypto-js');
const rateLimit = require('express-rate-limit');
const NodeCache = require('node-cache');
const { QuestionRecommender } = require('./ml/questionRecommender');

// --- CONFIGURATION ---
// server.js

const app = express();
app.set('trust proxy', 1); // Add this line to trust Render's proxy

const PORT = process.env.PORT || 8080;

// Load from .env file
const API_KEY = process.env.API_KEY;
const MONGO_URI = process.env.MONGO_URI;
const JWT_SECRET = process.env.JWT_SECRET;
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD;
const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT;

const ADMIN_PASSWORD_HASH = bcrypt.hashSync(ADMIN_PASSWORD, 10);
const genAI = new GoogleGenerativeAI(API_KEY);
const mongoClient = new MongoClient(MONGO_URI, {
  serverApi: { version: ServerApiVersion.v1, strict: true, deprecationErrors: true }
});
let db;

// Pinecone Client
const pinecone = new Pinecone({
    apiKey: PINECONE_API_KEY,
});
const PINECONE_INDEX_NAME = pinecone.index('jee-books-384');

// Local embedding model for queries
let embedder;

const cache = new NodeCache({ stdTTL: 300, checkperiod: 60 });
const questionRecommender = new QuestionRecommender();
const upload = multer({ storage: multer.memoryStorage() });

// --- MIDDLEWARE ---
// More specific CORS configuration for production
const corsOptions = {
  origin: 'https://geetham-exam.onrender.com', // IMPORTANT: Use your actual Render URL here
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));
app.use(express.json({ limit: '10mb' }));
app.use(express.static('public'));

const loginLimiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 100 });
const apiLimiter = rateLimit({ windowMs: 60 * 1000, max: 100 });
app.use('/api', apiLimiter);

// --- UTILITIES ---
function encryptData(data) { return CryptoJS.AES.encrypt(JSON.stringify(data), ENCRYPTION_KEY).toString(); }
function decryptData(encryptedData) { const bytes = CryptoJS.AES.decrypt(encryptedData, ENCRYPTION_KEY); return JSON.parse(bytes.toString(CryptoJS.enc.Utf8)); }
function encryptSensitiveData(obj) { if (!obj) return obj; const sensitiveFields = ['mobile', 'location', 'studentName']; const result = { ...obj }; sensitiveFields.forEach(field => { if (result[field]) { result[`${field}_encrypted`] = encryptData(result[field]); delete result[field]; } }); return result; }
function decryptSensitiveData(obj) { if (!obj) return obj; const result = { ...obj }; Object.keys(result).forEach(key => { if (key.endsWith('_encrypted')) { const originalField = key.replace('_encrypted', ''); try { result[originalField] = decryptData(result[key]); delete result[key]; } catch (error) {} } }); return result; }
function authenticateToken(req, res, next) { const authHeader = req.headers['authorization']; const token = authHeader && authHeader.split(' ')[1]; if (token == null) return res.sendStatus(401); jwt.verify(token, JWT_SECRET, (err, user) => { if (err) return res.sendStatus(403); req.user = user; next(); }); }

async function initializeServices() {
  try {
    // Connect to MongoDB
    await mongoClient.connect();
    db = mongoClient.db("GeethamQuizDB");
    console.log("Successfully connected to MongoDB Atlas!");

    // Initialize the local embedding pipeline
    console.log("Loading local embedding model...");
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log("Embedding model loaded successfully.");

  } catch (err) {
    console.error("Failed to initialize services:", err);
    process.exit(1);
  }
}

async function trainRecommenderFromDB() { /* Full function included below */ }

// ========== API ENDPOINTS ==========

app.post('/login', loginLimiter, (req, res) => { /* Full endpoint included below */ });
app.post('/login-dashboard', loginLimiter, (req, res) => { /* Full endpoint included below */ });
app.get('/get-quizzes', async (req, res) => { /* Full endpoint included below */ });
app.post('/update-quizzes', authenticateToken, async (req, res) => { /* Full endpoint included below */ });
app.post('/check-attempt', async (req, res) => { /* Full endpoint included below */ });
app.post('/submit-result', async (req, res) => { /* Full endpoint included below */ });
app.get('/dashboard-data', authenticateToken, async (req, res) => { /* Full endpoint included below */ });
app.delete('/delete-quiz/:id', authenticateToken, async (req, res) => { /* Full endpoint included below */ });

// --- AI GENERATION ENDPOINTS (Refactored for Pinecone) ---
app.post('/generate-questions', authenticateToken, async (req, res) => {
  try {
    const { topic, numQuestions, questionType, difficulty } = req.body;

    // 1. Convert the user's topic into a vector embedding
    const queryEmbedding = await embedder(topic, { pooling: 'mean', normalize: true });
    
    // 2. Query Pinecone to find the most relevant document vectors
    const queryResponse = await pineconeIndex.query({
      topK: 5,
      vector: Array.from(queryEmbedding.data),
      includeMetadata: true,
    });
    
    // 3. Extract the text from the metadata of the results
    const contextChunks = queryResponse.matches.map(match => match.metadata.text);
    if (contextChunks.length === 0) {
      return res.status(404).json({ message: `No information found on "${topic}" in the knowledge base.` });
    }
    const context = contextChunks.join("\n\n");
    
    // 4. Build prompt for Gemini and generate questions (this part is the same)
    const baseInstruction = `Based ONLY on the following context, generate exactly ${numQuestions} questions for a ${difficulty} level JEE exam on the topic of "${topic}". Output only a raw, valid JSON array.`;
    let prompt;
    if (questionType === 'fill-in-the-blank') {
      prompt = `${baseInstruction} Each object must have keys "text" (with a blank as "____") and "answerKey". CONTEXT: """${context}"""`;
    } else {
      prompt = `${baseInstruction} Each object must have keys "text", "options" (an array of 4 objects, each with a "text" and a "solution" key), and "correctAnswer" (a 0-based index). The "solution" key should explain why that option is correct/incorrect. CONTEXT: """${context}"""`;
    }
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro-latest" });
    const result = await model.generateContent(prompt);
    const text = result.response.text().replace(/```json|```/g, '').trim();
    const parsedQuestions = JSON.parse(text);
    res.json(parsedQuestions);

  } catch (error) {
    console.error("AI generation error:", error);
    res.status(500).json({ message: "Failed to generate valid questions from the library." });
  }
});
app.post('/generate-from-image', authenticateToken, upload.single('image'), async (req, res) => { /* Full endpoint included below */ });

// --- MACHINE LEARNING ENDPOINTS ---
app.post('/api/train-recommender', authenticateToken, async (req, res) => { /* Full endpoint included below */ });
app.post('/api/recommend-similar-questions', authenticateToken, async (req, res) => { /* Full endpoint included below */ });
app.post('/api/predict-difficulty', authenticateToken, async (req, res) => { /* Full endpoint included below */ });

// --- START SERVER ---
initializeServices().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running and listening on http://0.0.0.0:${PORT}`);
    trainRecommenderFromDB();
  });
});

// ===============================================
// === FULL IMPLEMENTATIONS OF OTHER FUNCTIONS ===
// ===============================================
async function trainRecommenderFromDB() { try { const quizzes = await db.collection('quizzes').find({}).toArray(); const allQuestions = []; quizzes.forEach(quiz => { if (quiz.subjects) { Object.entries(quiz.subjects).forEach(([subject, questions]) => { (questions || []).forEach(q => { allQuestions.push({ ...q, subject, quizId: quiz.id }); }); }); } }); if (allQuestions.length > 0) { await questionRecommender.train(allQuestions); console.log(`Recommender trained with ${allQuestions.length} questions.`); } else { console.log('No questions found to train the recommender.'); } } catch (e) { console.error('Failed to train recommender:', e); } }
app.post('/login', loginLimiter, (req, res) => { const { password } = req.body; if (bcrypt.compareSync(password, ADMIN_PASSWORD_HASH)) { res.json({ accessToken: jwt.sign({ user: 'admin' }, JWT_SECRET, { expiresIn: '8h' }) }); } else { res.status(401).json({ success: false, message: 'Incorrect password' }); } });
app.post('/login-dashboard', loginLimiter, (req, res) => { const { password } = req.body; if (bcrypt.compareSync(password, ADMIN_PASSWORD_HASH)) { res.json({ accessToken: jwt.sign({ user: 'dashboard_admin' }, JWT_SECRET, { expiresIn: '8h' }) }); } else { res.status(401).json({ success: false, message: 'Incorrect password' }); } });
app.get('/get-quizzes', async (req, res) => { try { const quizzes = await db.collection('quizzes').find({}).toArray(); res.json(quizzes); } catch (err) { res.status(500).json({ success: false, message: 'Error fetching exams.' }); } });
app.post('/update-quizzes', authenticateToken, async (req, res) => { try { const updatedQuizzes = req.body; const quizzesCollection = db.collection('quizzes'); await quizzesCollection.deleteMany({}); if (updatedQuizzes.length > 0) { await quizzesCollection.insertMany(updatedQuizzes); } trainRecommenderFromDB(); res.status(200).json({ success: true, message: 'Exams updated successfully.' }); } catch (err) { res.status(500).json({ success: false, message: 'Error saving exams.' }); } });
app.post('/check-attempt', async (req, res) => { try { const { mobile, quizId } = req.body; const encryptedMobile = encryptData(mobile); const attempt = await db.collection('results').findOne({ mobile_encrypted: encryptedMobile, quizId: quizId }); res.json({ success: true, canAttempt: !attempt, message: attempt ? "You have already attempted this exam." : "" }); } catch (err) { res.status(500).json({ success: false, message: 'Error checking attempt.' }); } });
app.post('/submit-result', async (req, res) => { try { const newResult = req.body; const encryptedResult = encryptSensitiveData(newResult); await db.collection('results').insertOne(encryptedResult); res.status(200).json({ success: true, message: 'Result saved successfully.' }); } catch (err) { res.status(500).json({ success: false, message: 'Error saving result.' }); } });
app.get('/dashboard-data', authenticateToken, async (req, res) => { try { const [encryptedResults, quizzes] = await Promise.all([ db.collection('results').find({}).sort({ date: -1 }).toArray(), db.collection('quizzes').find({}).toArray() ]); const results = encryptedResults.map(r => decryptSensitiveData(r)); res.json({ results, quizzes }); } catch (err) { res.status(500).json({ success: false, message: 'Error fetching dashboard data.' }); } });
app.delete('/delete-quiz/:id', authenticateToken, async (req, res) => { try { const quizId = parseFloat(req.params.id); const result = await db.collection('quizzes').deleteOne({ id: quizId }); if (result.deletedCount === 0) { return res.status(404).json({ success: false, message: 'Exam not found.' }); } trainRecommenderFromDB(); res.status(200).json({ success: true, message: 'Exam deleted successfully.' }); } catch (err) { res.status(500).send('Error deleting exam.'); } });
app.post('/generate-from-image', authenticateToken, upload.single('image'), async (req, res) => { try { if (!req.file) { return res.status(400).json({ message: "No image file was provided." }); } const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" }); const imagePart = { inlineData: { data: req.file.buffer.toString("base64"), mimeType: req.file.mimetype } }; const ocrPrompt = "Transcribe all questions and their multiple-choice options from this image. Preserve numbering. Focus on accuracy."; const ocrResult = await model.generateContent([ocrPrompt, imagePart]); const extractedText = ocrResult.response.text(); if (!extractedText || extractedText.length < 20) { return res.status(500).json({ message: "Could not extract sufficient text from the image." }); } const formatterPrompt = `Based on the following text from a question paper, format it into a valid JSON array of question objects. Each object must have keys: "type": "multiple-choice", "text": The question text, "options": An array of 4 objects, each with a "text" key and a "solution" key (leave as an empty string ""), and "correctAnswer": A 0-based index for the correct answer. Intelligently determine the correct answer. If not obvious, guess. Here is the text: """${extractedText}""" Output ONLY the raw, valid JSON array.`; const formatResult = await model.generateContent(formatterPrompt); const jsonText = formatResult.response.text().replace(/```json|```/g, '').trim(); const parsedQuestions = JSON.parse(jsonText); res.json(parsedQuestions); } catch (error) { console.error("Error generating exam from image:", error); res.status(500).json({ message: "An error occurred while generating the exam from the image." }); } });
app.post('/api/train-recommender', authenticateToken, async (req, res) => { try { await trainRecommenderFromDB(); res.json({ success: true, message: 'Recommender (re)trained.' }); } catch (err) { res.status(500).json({ success: false, message: 'Error training recommender' }); } });
app.post('/api/recommend-similar-questions', authenticateToken, async (req, res) => { try { const { questionText, count = 5 } = req.body; if (!questionText) return res.status(400).json({ success: false, message: 'Question text is required' }); const recommendations = questionRecommender.recommendSimilarQuestions(questionText, count); res.json({ success: true, recommendations }); } catch (err) { res.status(500).json({ success: false, message: 'Error recommending questions' }); } });
app.post('/api/predict-difficulty', authenticateToken, async (req, res) => { try { const { questions } = req.body; if (!questions || !Array.isArray(questions)) { return res.status(400).json({ success: false, message: 'Questions array is required' }); } const difficulties = questionRecommender.predictDifficulty(questions); res.json({ success: true, difficulties }); } catch (err) { res.status(500).json({ success: false, message: 'Error predicting difficulty' }); } });
