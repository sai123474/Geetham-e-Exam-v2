const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

// Mock JWT and bcrypt
jest.mock('jsonwebtoken');
jest.mock('bcryptjs');

describe('Authentication', () => {
  const JWT_SECRET = "Geetham_e_exam2025";
  const ADMIN_PASSWORD = "Geetham@2014";
  const ADMIN_PASSWORD_HASH = "$2a$10$someHashedPassword";
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
  });

  test('Password hashing works correctly', () => {
    bcrypt.hashSync.mockReturnValue(ADMIN_PASSWORD_HASH);
    
    const hashedPassword = bcrypt.hashSync(ADMIN_PASSWORD, 10);
    
    expect(hashedPassword).toBe(ADMIN_PASSWORD_HASH);
    expect(bcrypt.hashSync).toHaveBeenCalledWith(ADMIN_PASSWORD, 10);
  });

  test('Password comparison works correctly - valid password', () => {
    bcrypt.compareSync.mockReturnValue(true);
    
    const isValid = bcrypt.compareSync(ADMIN_PASSWORD, ADMIN_PASSWORD_HASH);
    
    expect(isValid).toBe(true);
    expect(bcrypt.compareSync).toHaveBeenCalledWith(ADMIN_PASSWORD, ADMIN_PASSWORD_HASH);
  });

  test('Password comparison works correctly - invalid password', () => {
    bcrypt.compareSync.mockReturnValue(false);
    
    const isValid = bcrypt.compareSync('wrong-password', ADMIN_PASSWORD_HASH);
    
    expect(isValid).toBe(false);
    expect(bcrypt.compareSync).toHaveBeenCalledWith('wrong-password', ADMIN_PASSWORD_HASH);
  });

  test('JWT token generation works correctly', () => {
    const mockToken = 'mock-jwt-token';
    jwt.sign.mockReturnValue(mockToken);
    
    const token = jwt.sign({ user: 'admin' }, JWT_SECRET, { expiresIn: '8h' });
    
    expect(token).toBe(mockToken);
    expect(jwt.sign).toHaveBeenCalledWith({ user: 'admin' }, JWT_SECRET, { expiresIn: '8h' });
  });

  test('JWT token verification works correctly - valid token', () => {
    const mockDecodedToken = { user: 'admin' };
    jwt.verify.mockImplementation((token, secret, callback) => {
      callback(null, mockDecodedToken);
    });
    
    jwt.verify('valid-token', JWT_SECRET, (err, decoded) => {
      expect(err).toBeNull();
      expect(decoded).toEqual(mockDecodedToken);
    });
    
    expect(jwt.verify).toHaveBeenCalledWith('valid-token', JWT_SECRET, expect.any(Function));
  });

  test('JWT token verification works correctly - invalid token', () => {
    const mockError = new Error('Invalid token');
    jwt.verify.mockImplementation((token, secret, callback) => {
      callback(mockError, null);
    });
    
    jwt.verify('invalid-token', JWT_SECRET, (err, decoded) => {
      expect(err).toEqual(mockError);
      expect(decoded).toBeNull();
    });
    
    expect(jwt.verify).toHaveBeenCalledWith('invalid-token', JWT_SECRET, expect.any(Function));
  });
});