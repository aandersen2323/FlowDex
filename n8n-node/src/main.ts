import type { IExecuteFunctions } from 'n8n-workflow';
import { INodeType, INodeTypeDescription, NodeOperationError } from 'n8n-workflow';
import fetch from 'node-fetch';

export class FlowDex implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'FlowDex',
    name: 'flowDex',
    group: ['transform'],
    version: 1,
    description: 'Token-efficient LLM router',
    defaults: { name: 'FlowDex' },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      { displayName: 'API Base URL', name: 'apiBase', type: 'string', default: 'http://localhost:8787' },
      { displayName: 'Task', name: 'task', type: 'string', default: 'general' },
      { displayName: 'User Input', name: 'userInput', type: 'string', default: '' },
      { displayName: 'System Prompt', name: 'systemPrompt', type: 'string', default: '' },
      { displayName: 'Context IDs (comma separated)', name: 'contextIds', type: 'string', default: '' },
      { displayName: 'Tool Candidates (comma)', name: 'tools', type: 'string', default: '' },
      { displayName: 'Model', name: 'model', type: 'string', default: 'anthropic/claude-3-5-sonnet' }
    ],
  };

  async execute(this: IExecuteFunctions) {
    const apiBase = this.getNodeParameter('apiBase', 0) as string;
    const task = this.getNodeParameter('task', 0) as string;
    const userInput = this.getNodeParameter('userInput', 0) as string;
    const systemPrompt = this.getNodeParameter('systemPrompt', 0) as string;
    const contextIdsStr = this.getNodeParameter('contextIds', 0) as string;
    const toolsStr = this.getNodeParameter('tools', 0) as string;
    const model = this.getNodeParameter('model', 0) as string;

    const context_ids = contextIdsStr ? contextIdsStr.split(',').map(s => s.trim()).filter(Boolean) : [];
    const tool_candidates = toolsStr ? toolsStr.split(',').map(s => s.trim()).filter(Boolean) : [];

    try {
      const res = await fetch(`${apiBase}/infer`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          task, user_input: userInput, system_prompt: systemPrompt,
          context_ids, tool_candidates, model
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return this.helpers.returnJsonArray([{ flowdex: data }]);
    } catch (err: any) {
      throw new NodeOperationError(this.getNode(), err);
    }
  }
}
