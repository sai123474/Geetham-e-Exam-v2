const request = require('supertest');
const { MongoClient } = require('mongodb');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

// Mock MongoDB
jest.mock('mongodb');

// Import server after mocking dependencies
let app;
let mockDb;
let mockCollection;

describe('API Endpoints', () => {
  beforeAll(() => {
    // Mock MongoDB client and collections
    mockCollection = {
      find: jest.fn().mockReturnThis(),
      findOne: jest.fn(),
      insertOne: jest.fn(),
      insertMany: jest.fn(),
      deleteMany: jest.fn(),
      deleteOne: jest.fn(),
      updateOne: jest.fn(),
      toArray: jest.fn()
    };
    
    mockDb = {
      collection: jest.fn().mockReturnValue(mockCollection)
    };
    
    MongoClient.connect = jest.fn().mockResolvedValue({
      db: jest.fn().mockReturnValue(mockDb)
    });
    
    // Mock JWT
    jwt.verify = jest.fn((token, secret, callback) => {
      callback(null, { user: 'admin' });
    });
    
    // Mock bcrypt
    bcrypt.compareSync = jest.fn().mockReturnValue(true);
    
    // Now import the server
    app = require('../server');
  });

  afterAll(async () => {
    await new Promise(resolve => setTimeout(() => resolve(), 500)); // avoid jest open handle error
  });

  // Test login endpoint
  test('POST /login - success', async () => {
    const response = await request(app)
      .post('/login')
      .send({ password: 'Geetham@2014' });
    
    expect(response.statusCode).toBe(200);
    expect(response.body).toHaveProperty('accessToken');
  });

  // Test get-quizzes endpoint
  test('GET /get-quizzes - success', async () => {
    mockCollection.toArray.mockResolvedValueOnce([
      { id: 1, title: 'Test Quiz 1' },
      { id: 2, title: 'Test Quiz 2' }
    ]);
    
    const response = await request(app)
      .get('/get-quizzes');
    
    expect(response.statusCode).toBe(200);
    expect(Array.isArray(response.body)).toBeTruthy();
    expect(response.body.length).toBe(2);
  });

  // Test check-attempt endpoint
  test('POST /check-attempt - success', async () => {
    mockCollection.findOne.mockResolvedValueOnce(null); // No previous attempt
    
    const response = await request(app)
      .post('/check-attempt')
      .send({ mobile: '1234567890', quizId: 1 });
    
    expect(response.statusCode).toBe(200);
    expect(response.body).toHaveProperty('canAttempt', true);
  });

  // Test submit-result endpoint
  test('POST /submit-result - success', async () => {
    mockCollection.insertOne.mockResolvedValueOnce({ acknowledged: true });
    
    const response = await request(app)
      .post('/submit-result')
      .send({
        studentName: 'Test Student',
        mobile: '1234567890',
        quizId: 1,
        totalScore: 80,
        subjectScores: { 'Math': 40, 'Science': 40 }
      });
    
    expect(response.statusCode).toBe(200);
    expect(response.body).toHaveProperty('success', true);
  });

  // Test clear-results endpoint with authentication
  test('DELETE /clear-results - success with auth', async () => {
    mockCollection.deleteMany.mockResolvedValueOnce({ deletedCount: 10 });
    
    const response = await request(app)
      .delete('/clear-results')
      .set('Authorization', 'Bearer fake-token');
    
    expect(response.statusCode).toBe(200);
    expect(response.body).toHaveProperty('deletedCount', 10);
  });

  // Test delete-quiz endpoint with authentication
  test('DELETE /delete-quiz/:id - success with auth', async () => {
    mockCollection.deleteOne.mockResolvedValueOnce({ deletedCount: 1 });
    
    const response = await request(app)
      .delete('/delete-quiz/1')
      .set('Authorization', 'Bearer fake-token');
    
    expect(response.statusCode).toBe(200);
    expect(response.body || response.text).toEqual(expect.anything());
  });
});