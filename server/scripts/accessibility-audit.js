/**
 * Accessibility Audit Script
 * 
 * This script performs an accessibility audit on the HTML files in the project
 * using the axe-core library to check for WCAG compliance.
 */

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const { default: axe } = require('axe-core');

// HTML files to audit
const filesToAudit = [
  'index (1).html',
  'admin.html',
  'results.html',
  'analytics.html'
];

// WCAG compliance levels to check
const tags = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];

// Function to run the accessibility audit
async function runAccessibilityAudit(filePath) {
  console.log(`\n🔍 Auditing ${filePath}...`);
  
  try {
    // Read the HTML file
    const html = fs.readFileSync(path.resolve(__dirname, '..', filePath), 'utf8');
    
    // Create a virtual DOM
    const { window } = new JSDOM(html, {
      resources: 'usable',
      runScripts: 'dangerously'
    });
    
    // Wait for any scripts to load
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Configure axe
    axe.configure({
      reporter: 'v2',
      checks: [
        { id: 'color-contrast', options: { noScroll: true } }
      ]
    });
    
    // Run the audit
    const results = await axe.run(window.document, {
      rules: {
        'color-contrast': { enabled: true }
      },
      resultTypes: ['violations', 'incomplete'],
      tags
    });
    
    // Process results
    const { violations, incomplete } = results;
    
    if (violations.length === 0 && incomplete.length === 0) {
      console.log('✅ No accessibility issues found!');
      return { file: filePath, violations: [], incomplete: [], passed: true };
    }
    
    // Report violations
    if (violations.length > 0) {
      console.log(`\n❌ Found ${violations.length} accessibility violations:`);
      violations.forEach((violation, index) => {
        console.log(`\n${index + 1}. ${violation.id}: ${violation.help}`);
        console.log(`   Impact: ${violation.impact}`);
        console.log(`   Description: ${violation.description}`);
        console.log(`   WCAG: ${violation.tags.filter(tag => tag.startsWith('wcag')).join(', ')}`);
        console.log(`   Elements affected: ${violation.nodes.length}`);
        
        violation.nodes.forEach((node, nodeIndex) => {
          console.log(`   - Element ${nodeIndex + 1}: ${node.html}`);
          console.log(`     Fix: ${node.failureSummary}`);
        });
      });
    }
    
    // Report incomplete checks
    if (incomplete.length > 0) {
      console.log(`\n⚠️ Found ${incomplete.length} incomplete checks that need manual review:`);
      incomplete.forEach((check, index) => {
        console.log(`\n${index + 1}. ${check.id}: ${check.help}`);
        console.log(`   Description: ${check.description}`);
        console.log(`   Elements affected: ${check.nodes.length}`);
      });
    }
    
    return {
      file: filePath,
      violations,
      incomplete,
      passed: false
    };
    
  } catch (error) {
    console.error(`Error auditing ${filePath}:`, error);
    return {
      file: filePath,
      error: error.message,
      passed: false
    };
  }
}

// Main function to run audits on all files
async function runAllAudits() {
  console.log('🚀 Starting accessibility audit...');
  
  const results = [];
  
  for (const file of filesToAudit) {
    const result = await runAccessibilityAudit(file);
    results.push(result);
  }
  
  // Generate summary
  console.log('\n📊 Audit Summary:');
  const passedFiles = results.filter(r => r.passed).length;
  console.log(`Files passed: ${passedFiles}/${results.length}`);
  
  const totalViolations = results.reduce((sum, r) => sum + (r.violations?.length || 0), 0);
  console.log(`Total violations: ${totalViolations}`);
  
  const totalIncomplete = results.reduce((sum, r) => sum + (r.incomplete?.length || 0), 0);
  console.log(`Total incomplete checks: ${totalIncomplete}`);
  
  // Generate report file
  const report = {
    summary: {
      date: new Date().toISOString(),
      filesAudited: results.length,
      filesPassed: passedFiles,
      totalViolations,
      totalIncomplete
    },
    results
  };
  
  fs.writeFileSync(
    path.resolve(__dirname, '..', 'accessibility-report.json'),
    JSON.stringify(report, null, 2)
  );
  
  console.log('\n📝 Report saved to accessibility-report.json');
  
  return report;
}

// Run the audits
runAllAudits().catch(console.error);