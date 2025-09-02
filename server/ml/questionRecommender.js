
/**
 * Question Recommender System
 * 
 * This module provides machine learning-based question recommendation functionality
 * for the Geetham e-Exam platform.
 */

const tf = require('@tensorflow/tfjs-node');

// TF-IDF Vectorizer for text processing
class TFIDFVectorizer {
  constructor() {
    this.vocabulary = {};
    this.idf = {};
    this.fitted = false;
  }

  // Tokenize text into words
  tokenize(text) {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 2);
  }

  // Fit the vectorizer on a corpus of texts
  fit(texts) {
    const docFrequency = {};
    const numDocs = texts.length;
    
    // Count document frequency for each term
    texts.forEach(text => {
      const tokens = this.tokenize(text);
      const uniqueTokens = [...new Set(tokens)];
      
      uniqueTokens.forEach(token => {
        docFrequency[token] = (docFrequency[token] || 0) + 1;
        this.vocabulary[token] = this.vocabulary[token] || Object.keys(this.vocabulary).length;
      });
    });
    
    // Calculate IDF for each term
    Object.keys(docFrequency).forEach(term => {
      this.idf[term] = Math.log(numDocs / (docFrequency[term] + 1)) + 1;
    });
    
    this.fitted = true;
    return this;
  }

  // Transform texts to TF-IDF vectors
  transform(texts) {
    if (!this.fitted) {
      throw new Error('Vectorizer must be fitted before transform');
    }
    
    const numFeatures = Object.keys(this.vocabulary).length;
    const vectors = [];
    
    texts.forEach(text => {
      const vector = new Array(numFeatures).fill(0);
      const tokens = this.tokenize(text);
      
      // Count term frequency
      const termFreq = {};
      tokens.forEach(token => {
        termFreq[token] = (termFreq[token] || 0) + 1;
      });
      
      // Calculate TF-IDF for each term
      Object.keys(termFreq).forEach(term => {
        if (this.vocabulary[term] !== undefined) {
          const tf = termFreq[term] / tokens.length;
          const tfidf = tf * (this.idf[term] || 1);
          vector[this.vocabulary[term]] = tfidf;
        }
      });
      
      vectors.push(vector);
    });
    
    return vectors;
  }

  // Fit and transform in one step
  fitTransform(texts) {
    return this.fit(texts).transform(texts);
  }
}

// Question Recommender class
class QuestionRecommender {
  constructor() {
    this.vectorizer = new TFIDFVectorizer();
    this.questionVectors = [];
    this.questions = [];
    this.difficultyModel = null;
  }

  // Train the recommender on a set of questions
  async train(questions) {
    this.questions = questions;
    
    // Extract text features for content-based filtering
    const texts = questions.map(q => q.text);
    this.questionVectors = this.vectorizer.fitTransform(texts);
    
    // Train difficulty prediction model if we have enough data
    if (questions.length >= 20) {
      await this.trainDifficultyModel(questions);
    }
    
    return this;
  }

  // Train a model to predict question difficulty
  async trainDifficultyModel(questions) {
    // Extract features: question length, word count, etc.
    const features = questions.map(q => {
      const text = q.text;
      const wordCount = text.split(/\s+/).length;
      const charCount = text.length;
      const avgWordLength = charCount / wordCount;
      
      return [wordCount, charCount, avgWordLength];
    });
    
    // Extract labels: convert difficulty to numeric
    const difficultyMap = { 'Easy': 0, 'Medium': 1, 'Hard': 2 };
    const labels = questions.map(q => {
      // If difficulty is not specified, estimate based on correct answer rate
      if (!q.difficulty) {
        if (q.correctPercent >= 70) return 0; // Easy
        if (q.correctPercent >= 40) return 1; // Medium
        return 2; // Hard
      }
      return difficultyMap[q.difficulty] || 1;
    });
    
    // Create and train the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 8, activation: 'relu', inputShape: [3] }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    
    model.compile({
      optimizer: tf.train.adam(),
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    const xs = tf.tensor2d(features);
    const ys = tf.tensor1d(labels, 'int32');
    
    await model.fit(xs, ys, {
      epochs: 50,
      batchSize: 8,
      shuffle: true,
      verbose: 0
    });
    
    this.difficultyModel = model;
    return model;
  }

  // Predict difficulty of new questions
  predictDifficulty(questions) {
    if (!this.difficultyModel) {
      return questions.map(() => 'Medium'); // Default if no model
    }
    
    const features = questions.map(q => {
      const text = q.text;
      const wordCount = text.split(/\s+/).length;
      const charCount = text.length;
      const avgWordLength = charCount / wordCount;
      
      return [wordCount, charCount, avgWordLength];
    });
    
    const xs = tf.tensor2d(features);
    const predictions = this.difficultyModel.predict(xs);
    const difficultyLabels = ['Easy', 'Medium', 'Hard'];
    
    return Array.from(predictions.argMax(1).dataSync()).map(idx => difficultyLabels[idx]);
  }

  // Calculate cosine similarity between vectors
  cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    
    if (normA === 0 || normB === 0) {
      return 0;
    }
    
    return dotProduct / (normA * normB);
  }

  // Recommend questions similar to a given question
  recommendSimilarQuestions(questionText, count = 5) {
    // Vectorize the input question
    const queryVector = this.vectorizer.transform([questionText])[0];
    
    // Calculate similarity with all questions
    const similarities = this.questionVectors.map((vector, index) => ({
      index,
      similarity: this.cosineSimilarity(queryVector, vector)
    }));
    
    // Sort by similarity (descending) and return top matches
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, count)
      .map(item => this.questions[item.index]);
  }

  // Recommend questions based on student performance
  recommendPersonalizedQuestions(studentResponses, count = 5) {
    // Identify weak areas based on incorrect answers
    const subjectScores = {};
    let totalQuestions = 0;
    let correctAnswers = 0;
    
    Object.entries(studentResponses).forEach(([subject, questions]) => {
      subjectScores[subject] = { correct: 0, total: 0 };
      
      Object.values(questions).forEach(response => {
        if (response.status === 'answered') {
          subjectScores[subject].total++;
          totalQuestions++;
          
          if (response.isCorrect) {
            subjectScores[subject].correct++;
            correctAnswers++;
          }
        }
      });
    });
    
    // Calculate subject proficiency scores (0-1)
    const subjectProficiency = {};
    Object.entries(subjectScores).forEach(([subject, scores]) => {
      subjectProficiency[subject] = scores.total > 0 ? scores.correct / scores.total : 0.5;
    });
    
    // Calculate overall proficiency
    const overallProficiency = totalQuestions > 0 ? correctAnswers / totalQuestions : 0.5;
    
    // Filter questions by weak subjects and appropriate difficulty
    const recommendations = [];
    const weakSubjects = Object.entries(subjectProficiency)
      .filter(([_, score]) => score < overallProficiency)
      .map(([subject, _]) => subject);
    
    // If no weak subjects found, use all subjects
    const targetSubjects = weakSubjects.length > 0 ? weakSubjects : Object.keys(subjectProficiency);
    
    // Select questions from target subjects with appropriate difficulty
    this.questions.forEach(question => {
      if (targetSubjects.includes(question.subject)) {
        const subjectScore = subjectProficiency[question.subject] || 0.5;
        
        // Match difficulty to student proficiency
        let isAppropriate = false;
        if (subjectScore < 0.4 && question.difficulty === 'Easy') isAppropriate = true;
        else if (subjectScore >= 0.4 && subjectScore < 0.7 && question.difficulty === 'Medium') isAppropriate = true;
        else if (subjectScore >= 0.7 && question.difficulty === 'Hard') isAppropriate = true;
        
        if (isAppropriate) {
          recommendations.push(question);
        }
      }
    });
    
    // If not enough appropriate questions, add more questions
    if (recommendations.length < count) {
      const remainingCount = count - recommendations.length;
      const additionalQuestions = this.questions
        .filter(q => !recommendations.includes(q))
        .slice(0, remainingCount);
      
      recommendations.push(...additionalQuestions);
    }
    
    return recommendations.slice(0, count);
  }
}

module.exports = { QuestionRecommender, TFIDFVectorizer };
