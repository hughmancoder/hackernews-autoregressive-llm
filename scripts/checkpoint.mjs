try {
  const logPath = '../src/logs/llm-training.log'
  if (await fs.pathExists(logPath)) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
    const newLogPath = `../src/logs/llm-training_${timestamp}.log`
    await fs.move(logPath, newLogPath)
    console.log(`Moved existing log to: llm-training_${timestamp}.log`)
  }

  await $`git -C .. add .`

  const status = await $`git -C .. status --porcelain`
  if (status.stdout.trim() === '') {
    console.log('No changes to checkpoint')
    process.exit(0)
  }

  await $`git -C .. commit -m "llm-training auto checkpoint"`
  console.log('Auto checkpoint created')
} catch (error) {
  if (error.message.includes('nothing to commit')) {
    console.log('No changes to checkpoint')
    process.exit(0)
  }

  console.error('Failed to create checkpoint:', error.message)
  process.exit(1)
}