// server.js
const express = require('express');
const { MongoClient, ServerApiVersion, ObjectId } = require('mongodb');
const path = require('path');
const cors = require('cors');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const multer = require('multer');
const CryptoJS = require('crypto-js');
const rateLimit = require('express-rate-limit');
const NodeCache = require('node-cache');
const { ChromaClient } = require('chromadb');
const { QuestionRecommender } = require('./ml/questionRecommender');

// --- CONFIGURATION ---
const app = express();
const PORT = process.env.PORT || 3000;

// Load from .env file
const API_KEY = "AIzaSyA3WIHFw2_5E8hYUSlG2rrLdq7J7KKwbe0";
const MONGO_URI = "mongodb+srv://yamparalasaikrishna6:Tngy9EWjTg1akDXW@saikrishna.dced3fy.mongodb.net/?retryWrites=true&w=majority&appName=SaiKrishna";
const JWT_SECRET = "Geetham_e_exam2025";
const ADMIN_PASSWORD = "Geetham@2014";
const ENCRYPTION_KEY = "Geetham_secure_encryption_key_2025";

// Rate limiting
const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message: { success: false, message: 'Too many login attempts, please try again later.' },
});

const apiLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 60,
  message: { success: false, message: 'Too many requests, please try again later.' },
});

const ADMIN_PASSWORD_HASH = bcrypt.hashSync(ADMIN_PASSWORD, 10);
const genAI = new GoogleGenerativeAI(API_KEY);
const mongoClient = new MongoClient(MONGO_URI, {
  serverApi: { version: ServerApiVersion.v1, strict: true, deprecationErrors: true }
});
let db;

// ChromaDB Client for the pre-processed knowledge base
const chromaClient = new ChromaClient({ path: path.join(__dirname, 'db') });
let knowledgeCollection;

const cache = new NodeCache({ stdTTL: 300, checkperiod: 60 });
const questionRecommender = new QuestionRecommender();
const upload = multer({ storage: multer.memoryStorage() });

// --- MIDDLEWARE ---
app.use(cors());
app.use(express.json({ limit: '10mb' })); // Increase limit for image uploads
app.use(express.static('public'));
app.use('/api', apiLimiter);

// --- ENCRYPTION UTILITIES ---
function encryptData(data) {
  return CryptoJS.AES.encrypt(JSON.stringify(data), ENCRYPTION_KEY).toString();
}

function decryptData(encryptedData) {
  const bytes = CryptoJS.AES.decrypt(encryptedData, ENCRYPTION_KEY);
  return JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
}

function encryptSensitiveData(obj) {
  if (!obj) return obj;
  const sensitiveFields = ['mobile', 'location', 'studentName'];
  const result = { ...obj };
  sensitiveFields.forEach(field => {
    if (result[field]) {
      result[`${field}_encrypted`] = encryptData(result[field]);
      delete result[field];
    }
  });
  return result;
}

function decryptSensitiveData(obj) {
    if (!obj) return obj;
    const result = { ...obj };
    Object.keys(result).forEach(key => {
        if (key.endsWith('_encrypted')) {
            const originalField = key.replace('_encrypted', '');
            try {
                result[originalField] = decryptData(result[key]);
                delete result[key];
            } catch (error) {
                // Ignore decryption errors for fields that might not be encrypted
            }
        }
    });
    return result;
}

function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (token == null) return res.sendStatus(401);
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.sendStatus(403);
    req.user = user;
    next();
  });
}

async function connectDB() {
  try {
    await mongoClient.connect();
    db = mongoClient.db("GeethamQuizDB");
    await Promise.all([
      db.collection('quizzes').createIndex({ id: 1 }, { unique: true }),
      db.collection('results').createIndex({ mobile_encrypted: 1, quizId: 1 }),
      db.collection('results').createIndex({ date: -1 }),
    ]);
    console.log("Successfully connected to MongoDB Atlas!");
    
    knowledgeCollection = await chromaClient.getCollection({ name: "jee_books" });
    console.log(`Connected to ChromaDB. Found ${await knowledgeCollection.count()} items in knowledge base.`);

  } catch (err) {
    console.error("Failed to connect to databases", err);
    process.exit(1);
  }
}

async function trainRecommenderFromDB() {
  try {
    const quizzes = await db.collection('quizzes').find({}).toArray();
    const allQuestions = [];
    quizzes.forEach(quiz => {
      if (quiz.subjects) {
        Object.values(quiz.subjects).forEach(questions => {
          (questions || []).forEach(q => { allQuestions.push(q); });
        });
      }
    });
    if (allQuestions.length > 0) {
      await questionRecommender.train(allQuestions);
      console.log(`Recommender trained with ${allQuestions.length} questions.`);
    }
  } catch (e) {
    console.error('Failed to train recommender:', e);
  }
}

// ========== API ENDPOINTS ==========

app.post('/login', loginLimiter, (req, res) => {
  const { password } = req.body;
  if (bcrypt.compareSync(password, ADMIN_PASSWORD_HASH)) {
    res.json({ accessToken: jwt.sign({ user: 'admin' }, JWT_SECRET, { expiresIn: '4h' }) });
  } else {
    res.status(401).json({ success: false, message: 'Incorrect password' });
  }
});

app.post('/login-dashboard', loginLimiter, (req, res) => {
    const { password } = req.body;
    if (bcrypt.compareSync(password, ADMIN_PASSWORD_HASH)) {
        res.json({ accessToken: jwt.sign({ user: 'dashboard_admin' }, JWT_SECRET, { expiresIn: '4h' }) });
    } else {
        res.status(401).json({ success: false, message: 'Incorrect password' });
    }
});

app.get('/get-quizzes', async (req, res) => {
  try {
    const cachedQuizzes = cache.get('all_quizzes');
    if (cachedQuizzes) {
      return res.json(cachedQuizzes);
    }
    const quizzes = await db.collection('quizzes').find({}).toArray();
    cache.set('all_quizzes', quizzes);
    res.json(quizzes);
  } catch (err) {
    console.error('Error fetching quizzes:', err);
    res.status(500).json({ success: false, message: 'Error fetching quizzes.' });
  }
});

app.post('/update-quizzes', authenticateToken, async (req, res) => {
    try {
        const updatedQuizzes = req.body;
        if (!Array.isArray(updatedQuizzes)) {
            return res.status(400).json({ success: false, message: 'Invalid input: expected an array of quizzes.' });
        }
        const quizzesCollection = db.collection('quizzes');
        const session = mongoClient.startSession();
        try {
            await session.withTransaction(async () => {
                await quizzesCollection.deleteMany({}, { session });
                if (updatedQuizzes.length > 0) {
                    await quizzesCollection.insertMany(updatedQuizzes, { session });
                }
            });
        } finally {
            await session.endSession();
        }
        cache.del('all_quizzes');
        trainRecommenderFromDB();
        res.status(200).json({ success: true, message: 'Quizzes updated successfully.'});
    } catch (err) {
        console.error('Error saving quizzes:', err);
        res.status(500).json({ success: false, message: 'Error saving quizzes.' });
    }
});

app.post('/check-attempt', async (req, res) => {
  try {
    const { mobile, quizId } = req.body;
    if (!mobile || !quizId) {
      return res.status(400).json({ success: false, message: 'Missing required fields: mobile and quizId are required.' });
    }
    let attempt = await db.collection('results').findOne({ mobile, quizId: parseInt(quizId) });
    if (!attempt) {
      const quizResults = await db.collection('results').find({ quizId: parseInt(quizId) }).toArray();
      for (const result of quizResults) {
        const decrypted = decryptSensitiveData(result);
        if (decrypted.mobile === mobile) {
          attempt = decrypted;
          break;
        }
      }
    }
    res.json({ success: true, canAttempt: !attempt, message: attempt ? "You have already attempted this quiz." : "" });
  } catch (err) {
    console.error('Error checking attempt:', err);
    res.status(500).json({ success: false, message: 'Error checking attempt.' });
  }
});

app.post('/submit-result', async (req, res) => {
  try {
    const newResult = req.body;
    if (!newResult.studentName || !newResult.mobile || !newResult.quizId) {
      return res.status(400).json({ success: false, message: 'Missing required fields.' });
    }
    newResult.quizId = parseInt(newResult.quizId);
    if (!newResult.date) newResult.date = new Date().toISOString();
    const encryptedResult = encryptSensitiveData(newResult);
    await db.collection('results').insertOne(encryptedResult);
    cache.keys().forEach(key => key.startsWith('dashboard_') && cache.del(key));
    res.status(200).json({ success: true, message: 'Result saved successfully.' });
  } catch (err) {
    console.error('Error saving result:', err);
    res.status(500).json({ success: false, message: 'Error saving result.' });
  }
});

app.get('/dashboard-data', authenticateToken, async (req, res) => {
    try {
        const cacheKey = `dashboard_data`;
        const cachedData = cache.get(cacheKey);
        if (cachedData) {
            return res.json(cachedData);
        }
        const [encryptedResults, quizzes] = await Promise.all([
            db.collection('results').find({}).sort({ date: -1 }).toArray(),
            db.collection('quizzes').find({}).toArray()
        ]);
        const results = encryptedResults.map(r => decryptSensitiveData(r));
        const responseData = { results, quizzes };
        cache.set(cacheKey, responseData, 60);
        res.json(responseData);
    } catch (err) {
        console.error('Error fetching dashboard data:', err);
        res.status(500).json({ success: false, message: 'Error fetching dashboard data.' });
    }
});

app.delete('/delete-quiz/:id', authenticateToken, async (req, res) => {
  try {
    const quizId = parseInt(req.params.id) || req.params.id;
    const result = await db.collection('quizzes').deleteOne({ id: quizId });
    if (result.deletedCount === 0) {
      return res.status(404).json({ success: false, message: 'Quiz not found.' });
    }
    cache.del('all_quizzes');
    trainRecommenderFromDB();
    res.status(200).json({ success: true, message: 'Quiz deleted successfully.' });
  } catch (err) {
    console.error('Error deleting quiz:', err);
    res.status(500).send('Error deleting quiz.');
  }
});

// --- AI GENERATION ENDPOINTS ---

app.post('/generate-questions', authenticateToken, async (req, res) => {
  try {
    const { topic, numQuestions, questionType, difficulty } = req.body;
    const results = await knowledgeCollection.query({
        queryTexts: [topic], nResults: 5,
    });
    const contextChunks = results.documents[0];
    if (!contextChunks || contextChunks.length === 0) {
      return res.status(404).json({ message: `No information found on "${topic}" in the built-in library.` });
    }
    const context = contextChunks.join("\n\n");
    const baseInstruction = `Based ONLY on the following context, generate exactly ${numQuestions} questions for a ${difficulty} level JEE exam on the topic of "${topic}". Output only a raw, valid JSON array.`;
    let prompt;
    if (questionType === 'fill-in-the-blank') {
      prompt = `${baseInstruction} Each object must have keys "text" (with a blank as "____") and "answerKey". For answerKey, provide answers separated by '|'. CONTEXT: """${context}"""`;
    } else {
      prompt = `${baseInstruction} Each object must have keys "text", "options" (an array of 4 objects, each with a "text" and a "solution" key), and "correctAnswer" (a 0-based index). The "solution" key should contain a brief explanation for why that option is correct or incorrect. CONTEXT: """${context}"""`;
    }
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });
    const result = await model.generateContent(prompt);
    const text = result.response.text().replace(/```json|```/g, '').trim();
    const parsedQuestions = JSON.parse(text);
    res.json(parsedQuestions);
  } catch (error) {
    console.error("AI generation error:", error);
    res.status(500).json({ message: "Failed to generate valid questions from the library." });
  }
});

app.post('/generate-from-image', authenticateToken, upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ message: "No image file was provided." });
        }
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });
        const imagePart = {
            inlineData: { data: req.file.buffer.toString("base64"), mimeType: req.file.mimetype, },
        };
        const ocrPrompt = "Transcribe all the questions and their multiple-choice options from this image. Preserve the original numbering and lettering. Focus on accuracy.";
        const ocrResult = await model.generateContent([ocrPrompt, imagePart]);
        const extractedText = ocrResult.response.text();
        if (!extractedText || extractedText.length < 20) {
            return res.status(500).json({ message: "Could not extract sufficient text from the image." });
        }
        const formatterPrompt = `Based on the following text extracted from a question paper, format it into a valid JSON array of question objects. Each object must have keys: "type": "multiple-choice", "text": The question text., "options": An array of 4 objects, each with a "text" key (for the option) and a "solution" key (leave this as an empty string ""), and "correctAnswer": A 0-based index for the correct answer. You must intelligently determine the correct answer based on your knowledge. If the answer is not obvious, make your best educated guess. Here is the text: """ ${extractedText} """ Output ONLY the raw, valid JSON array.`;
        const formatResult = await model.generateContent(formatterPrompt);
        const jsonText = formatResult.response.text().replace(/```json|```/g, '').trim();
        const parsedQuestions = JSON.parse(jsonText);
        res.json(parsedQuestions);
    } catch (error) {
        console.error("Error generating quiz from image:", error);
        res.status(500).json({ message: "An error occurred while generating the quiz from the image." });
    }
});

// --- ML ENDPOINTS ---
app.post('/api/train-recommender', authenticateToken, async (req, res) => {
  try {
    await trainRecommenderFromDB();
    res.json({ success: true, message: 'Recommender (re)trained.' });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Error training recommender' });
  }
});
app.post('/api/recommend-similar-questions', authenticateToken, async (req, res) => {
  try {
    const { questionText, count = 5 } = req.body;
    if (!questionText) return res.status(400).json({ success: false, message: 'Question text is required' });
    const recommendations = questionRecommender.recommendSimilarQuestions(questionText, count);
    res.json({ success: true, recommendations });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Error recommending questions' });
  }
});

// --- START SERVER ---
connectDB().then(() => {
  app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
  trainRecommenderFromDB();
});