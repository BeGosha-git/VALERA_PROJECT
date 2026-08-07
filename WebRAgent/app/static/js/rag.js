// WebRAgent JavaScript functionality - Enhanced Modular Version

/**
 * WebRAgent Application
 * A modular and maintainable approach to WebRAgent interface functionality
 */
const WebRAgentApp = (function() {
    'use strict';

    // Private members
    const config = {
        selectors: {
            // Form elements
            queryForm: '#query-form',
            generateMindmap: '#generate-mindmap',
            useAgentSearch: '#use-agent-search',
            useWebSearch: '#use-web-search',
            useDeepSearch: '#use-deep-search',
            agentStrategy: '#agent-strategy',
            agentStrategyContainer: '.agent-strategy-container',
            collectionSelect: '#collection_id',
            collectionHelpText: '#collection-help',
            maxResults: '#max-results',
            responseContainer: '#response-container',
            loadingSpinner: '#loading-spinner',
            submitButton: 'button[type="submit"]',
            
            // Source type selection
            useCollectionOption: '#use-collection-option',
            useWebOption: '#use-web-option',
            collectionSelectorContainer: '#collection-selector-container',
            webSearchOptionsContainer: '#web-search-options-container',
            
            // Mind map elements
            mindmapContainer: '#mindmap-container',
            mindmapFrame: '#mindmap-frame',
            closeMindmapBtn: '#close-mindmap-btn',
            
            // Document preview elements
            documentPreviewModal: '#documentPreviewModal',
            modalPreviewIframe: '#modal-preview-iframe',
            modalPreviewDownload: '#modal-preview-download',
            modalPreviewTitle: '#documentPreviewModalLabel',
            
            // Admin elements
            deleteCollectionForms: '.delete-collection-form',
            deleteDocumentForms: '.delete-document-form',
            
            // Context links
            contextLinks: '.context-link'
        },
        endpoints: {
            query: '/query',
            mindmap: '/mindmap',
            documentRaw: '/document/%ID%/raw' 
        },
        storage: {
            generateMindmap: 'generate_mindmap',
            useAgentSearch: 'use_agent_search',
            useWebSearch: 'use_web_search',
            useDeepSearch: 'use_deep_search',
            agentStrategy: 'agent_strategy',
            maxResults: 'max_results',
            sourceTypeMain: 'source_type_main'
        },
        animations: {
            defaultDuration: 300,
            fadeIn: 'fade-in',
            slideIn: 'slide-in'
        }
    };
    
    let modal = null; // Will store Bootstrap modal instance
    
    /**
     * Preferences management for user settings
     */
    const Preferences = {
        save: function() {
            const generateMindmap = document.querySelector(config.selectors.generateMindmap);
            const useAgentSearch = document.querySelector(config.selectors.useAgentSearch);
            const useDeepSearch = document.querySelector(config.selectors.useDeepSearch);
            const agentStrategy = document.querySelector(config.selectors.agentStrategy);
            const maxResults = document.querySelector(config.selectors.maxResults);
            
            // Source type is stored separately when radio buttons are changed
            
            if (generateMindmap) {
                sessionStorage.setItem(config.storage.generateMindmap, generateMindmap.checked);
            }
            
            if (useAgentSearch) {
                sessionStorage.setItem(config.storage.useAgentSearch, useAgentSearch.checked);
            }
            
            if (useDeepSearch) {
                sessionStorage.setItem(config.storage.useDeepSearch, useDeepSearch.checked);
            }
            
            if (agentStrategy) {
                sessionStorage.setItem(config.storage.agentStrategy, agentStrategy.value);
            }
            
            if (maxResults) {
                sessionStorage.setItem(config.storage.maxResults, maxResults.value);
            }
        },
        
        load: function() {
            const generateMindmap = document.querySelector(config.selectors.generateMindmap);
            const useAgentSearch = document.querySelector(config.selectors.useAgentSearch);
            const useDeepSearch = document.querySelector(config.selectors.useDeepSearch);
            const agentStrategy = document.querySelector(config.selectors.agentStrategy);
            const maxResults = document.querySelector(config.selectors.maxResults);
            
            // Note: Source type preference (web vs collection) is loaded separately
            // in the radio button initialization code
            
            if (generateMindmap) {
                const preference = sessionStorage.getItem(config.storage.generateMindmap);
                if (preference !== null) {
                    generateMindmap.checked = preference === 'true';
                }
            }
            
            if (useAgentSearch) {
                const preference = sessionStorage.getItem(config.storage.useAgentSearch);
                if (preference !== null) {
                    useAgentSearch.checked = preference === 'true';
                    
                    // Trigger change event to update UI if needed
                    if (preference === 'true') {
                        const event = new Event('change');
                        useAgentSearch.dispatchEvent(event);
                    }
                }
            }
            
            if (useDeepSearch) {
                const preference = sessionStorage.getItem(config.storage.useDeepSearch);
                if (preference !== null) {
                    useDeepSearch.checked = preference === 'true';
                    
                    // Trigger change event to update UI if needed
                    if (preference === 'true') {
                        const event = new Event('change');
                        useDeepSearch.dispatchEvent(event);
                    }
                }
            }
            
            if (agentStrategy) {
                const value = sessionStorage.getItem(config.storage.agentStrategy);
                if (value !== null) {
                    agentStrategy.value = value;
                }
            }
            
            if (maxResults) {
                const value = sessionStorage.getItem(config.storage.maxResults);
                if (value !== null) {
                    maxResults.value = value;
                }
            }
        }
    };
    
    /**
     * Query form handling and result display
     */
    const QueryProcessor = {
        init: function() {
            const form = document.querySelector(config.selectors.queryForm);
            if (!form) return;
            
            form.addEventListener('submit', this.handleSubmit.bind(this));
            
            // Add listener for form submission animation (separate from AJAX handling)
            form.addEventListener('submit', function(e) {
                if (!e.defaultPrevented) {
                    const submitBtn = this.querySelector('button[type="submit"]');
                    if (submitBtn) {
                        submitBtn.disabled = true;
                        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                    }
                }
            });
        },
        
        handleSubmit: function(e) {
            e.preventDefault();
            
            // Update UI to show loading state
            this.showLoading();
            
            // Save user preferences
            Preferences.save();
            
            // Get form data
            const form = e.target;
            const formData = new FormData(form);
            
            // Send AJAX request
            fetch(config.endpoints.query, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Process successful response
                this.hideLoading();
                this.displayResults(data);
            })
            .catch(error => {
                // Handle error gracefully
                console.error('Error processing query:', error);
                this.handleError(error);
            });
        },
        
        showLoading: function() {
            // Hide empty results placeholder
            const emptyPlaceholder = document.getElementById('empty-results-placeholder');
            if (emptyPlaceholder) {
                emptyPlaceholder.style.display = 'none';
            }
            
            // Clear previous results
            const responseContainer = document.querySelector(config.selectors.responseContainer);
            if (responseContainer) {
                responseContainer.innerHTML = '';
                responseContainer.style.display = 'none';
            }
            
            // Hide sources container
            const sourcesContainer = document.getElementById('sources-container');
            if (sourcesContainer) {
                sourcesContainer.style.display = 'none';
            }
            
            // Show spinner
            const spinner = document.querySelector(config.selectors.loadingSpinner);
            if (spinner) {
                spinner.style.display = 'block';
            }
            
            // Update submit button
            const submitBtn = document.querySelector(config.selectors.submitButton);
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            }
            
            // Set processing message based on selected options
            this.updateProcessingMessage();
        },
        
        updateProcessingMessage: function() {
            const processingDetails = document.getElementById('processing-details');
            if (!processingDetails) return;
            
            // Get selected search method - now using radio buttons
            const useWebSearch = document.getElementById('use-web-option').checked;
            const useDeepSearch = document.getElementById('use-deep-search').checked;
            const useAgentSearch = document.getElementById('use-agent-search').checked;
            const collectionSelect = document.getElementById('collection_id');
            const agentStrategy = document.getElementById('agent-strategy');
            
            let statusMessages = [];
            
            // Determine what type of search we're doing
            if (useWebSearch) {
                if (useDeepSearch) {
                    if (useAgentSearch) {
                        // Deep search with agent capabilities
                        const strategyName = agentStrategy.value === 'informed' ? 'Informed Decomposition' : 'Direct Decomposition';
                        statusMessages.push('<i class="bi bi-globe me-1"></i>Searching the web for relevant information...');
                        statusMessages.push('<i class="bi bi-robot me-1"></i>Breaking down query using ' + strategyName + ' strategy...');
                        statusMessages.push('<i class="bi bi-search-heart me-1"></i>Scraping full content from pages for each subquery...');
                        statusMessages.push('<i class="bi bi-braces-asterisk me-1"></i>Analyzing each page with LLM for detailed extraction...');
                        statusMessages.push('<i class="bi bi-arrows-angle-contract me-1"></i>Synthesizing comprehensive answer from all subqueries...');
                    } else {
                        // Standard deep search
                        statusMessages.push('<i class="bi bi-globe me-1"></i>Searching the web for relevant information...');
                        statusMessages.push('<i class="bi bi-search-heart me-1"></i>Scraping full content from web pages...');
                        statusMessages.push('<i class="bi bi-braces-asterisk me-1"></i>Analyzing each page with LLM for detailed content extraction...');
                        statusMessages.push('<i class="bi bi-arrows-angle-contract me-1"></i>Synthesizing comprehensive answer from all sources...');
                    }
                } else {
                    statusMessages.push('<i class="bi bi-globe me-1"></i>Searching the web for relevant information...');
                    
                    if (useAgentSearch) {
                        const strategyName = agentStrategy.value === 'informed' ? 'Informed Decomposition' : 'Direct Decomposition';
                        statusMessages.push('<i class="bi bi-robot me-1"></i>Breaking down query using ' + strategyName + ' strategy...');
                        statusMessages.push('<i class="bi bi-arrows-angle-contract me-1"></i>Analyzing information and synthesizing response...');
                    }
                }
            } else if (collectionSelect && collectionSelect.selectedIndex > 0) {
                const collectionName = collectionSelect.options[collectionSelect.selectedIndex].text;
                statusMessages.push('<i class="bi bi-database-search me-1"></i>Searching in collection: ' + collectionName);
                
                if (useAgentSearch) {
                    const strategyName = agentStrategy.value === 'informed' ? 'Informed Decomposition' : 'Direct Decomposition';
                    statusMessages.push('<i class="bi bi-robot me-1"></i>Using agent search with ' + strategyName + ' strategy...');
                } else {
                    statusMessages.push('<i class="bi bi-vector-pen me-1"></i>Processing query through vector database...');
                }
                
                statusMessages.push('<i class="bi bi-cpu me-1"></i>Generating LLM response based on retrieved context...');
            }
            
            // Add LLM model information
            const activeProvider = localStorage.getItem('active_llm_provider') || 'Unknown';
            const activeModel = localStorage.getItem('active_llm_model') || 'Default model';
            statusMessages.push(`<i class="bi bi-lightning-charge me-1"></i>Using ${activeProvider} (${activeModel})...`);
            
            // Display status messages
            if (statusMessages.length > 0) {
                processingDetails.innerHTML = statusMessages.map(msg => `<div class="mb-1">${msg}</div>`).join('');
            } else {
                processingDetails.innerHTML = '<div class="mb-1"><i class="bi bi-hourglass-split me-1"></i>Processing your request...</div>';
            }
        },
        
        hideLoading: function() {
            // Hide spinner
            const spinner = document.querySelector(config.selectors.loadingSpinner);
            if (spinner) {
                spinner.style.display = 'none';
            }
            
            // Show response container
            const responseContainer = document.querySelector(config.selectors.responseContainer);
            if (responseContainer) {
                responseContainer.style.display = 'block';
            }
            
            // Reset submit button
            const submitBtn = document.querySelector(config.selectors.submitButton);
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<span class="spinner-submit"><i class="bi bi-send me-1"></i>Submit Query</span>';
            }
        },
        
        displayResults: function(data) {
            const responseContainer = document.querySelector(config.selectors.responseContainer);
            if (!responseContainer) return;
            
            // Update the results header title with query using the specific ID
            const resultsCard = document.getElementById('results-card');
            if (resultsCard && data.query) {
                const headerTitle = resultsCard.querySelector('.card-header-custom .card-title');
                if (headerTitle) {
                    headerTitle.innerHTML = '<i class="bi bi-search me-2"></i>Results for: <span class="text-primary">"' + data.query + '"</span>';
                }
            }
            
            // Store model information in localStorage if available
            if (data.model_info) {
                localStorage.setItem('active_llm_provider', data.model_info.provider || 'Unknown');
                localStorage.setItem('active_llm_model', data.model_info.model || 'Default model');
            }
            
            // Build HTML for the response section
            let html = this.buildResponseHTML(data);
            
            // Update the DOM
            responseContainer.innerHTML = html;
            responseContainer.style.display = 'block';
            
            // Handle sources container
            const sourcesContainer = document.getElementById('sources-container');
            if (sourcesContainer) {
                if (data.contexts && data.contexts.length > 0) {
                    // Build sources HTML
                    let sourcesHTML = this.buildSourcesHTML(data.contexts, data.search_type);
                    
                    // Update the sources section
                    const sourcesContent = sourcesContainer.querySelector('.source-documents');
                    if (sourcesContent) {
                        sourcesContent.innerHTML = sourcesHTML;
                    }
                    
                    // Update the header title based on search type
                    const sourcesHeader = sourcesContainer.querySelector('.card-header-custom .card-title');
                    if (sourcesHeader) {
                        if (data.search_type === 'web') {
                            sourcesHeader.innerHTML = '<i class="bi bi-globe me-2"></i>Web Sources';
                        } else {
                            sourcesHeader.innerHTML = '<i class="bi bi-journal-text me-2"></i>Source Documents';
                        }
                    }
                    
                    // Show the sources container
                    sourcesContainer.style.display = 'block';
                    
                    // Re-attach event listeners to context links
                    DocumentPreview.setupLinks();
                } else {
                    // Hide sources container if no sources
                    sourcesContainer.style.display = 'none';
                }
            }
            
            // Set up interactive elements in the response
            this.setupInteractions();
            
            // Handle mindmap generation if enabled
            const generateMindmapCheckbox = document.querySelector(config.selectors.generateMindmap);
            if (generateMindmapCheckbox && generateMindmapCheckbox.checked && data.contexts && data.contexts.length > 0) {
                MindmapHandler.generate(data.query, data.contexts);
            }
            
            // Scroll to the results using the specific ID
            const resultsCardForScroll = document.getElementById('results-card');
            if (resultsCardForScroll) {
                resultsCardForScroll.scrollIntoView({ behavior: 'smooth' });
            }
        },
        
        buildResponseHTML: function(data) {
            // Determine result type
            const isAgentSearch = Array.isArray(data.subqueries) && data.subqueries.length > 0;
            const isWebSearch = data.search_type === 'web';
            const agentStrategy = data.strategy || 'direct';

            let html = '';
            
            // Build appropriate header badges
            let badgeHTML = '';
            
            if (isWebSearch) {
                badgeHTML += `<span class="badge bg-primary text-white ms-2"><i class="bi bi-globe me-1"></i>Web Search</span>`;
            }
            
            if (isAgentSearch) {
                badgeHTML += `<span class="badge bg-info text-dark ms-2"><i class="bi bi-robot me-1"></i>Agent Search</span>`;
                
                const strategyText = agentStrategy === 'informed' ? 
                    'Informed Decomposition' : 'Direct Decomposition';
                badgeHTML += `<span class="badge bg-secondary text-white ms-1">${strategyText}</span>`;
            }
            
            // Add model info badge if available
            if (data.model_info) {
                badgeHTML += `<span class="badge bg-dark text-white ms-1"><i class="bi bi-lightning-charge me-1"></i>${data.model_info.provider} (${data.model_info.model})</span>`;
            }
            
            // Generate response content based on type
            if (isAgentSearch) {
                // Agent Search format with subqueries
                html = `
                    <h4 class="h5 fw-bold mb-2">
                        <i class="bi bi-chat-dots me-2"></i>Generated Response:
                    </h4>
                    <div class="badge-container mb-3">
                        ${badgeHTML}
                    </div>
                    
                    <div class="llm-response mb-4">
                        ${data.response}
                    </div>
                    
                    <div class="mb-3">
                        <div class="d-flex align-items-center mb-2">
                            <button class="btn btn-sm btn-outline-secondary me-2" type="button" data-bs-toggle="collapse" data-bs-target="#subqueriesCollapse" aria-expanded="false" aria-controls="subqueriesCollapse">
                                <i class="bi bi-list-task me-1"></i>Show Subqueries
                            </button>
                            <small class="text-muted">See how the question was broken down</small>
                        </div>
                        
                        <div class="collapse" id="subqueriesCollapse">
                            <div class="card card-body bg-light mb-3">
                                <h6 class="card-subtitle mb-2 text-muted">Query Decomposition:</h6>
                                <ul class="list-group list-group-flush">
                                    ${data.subqueries.map(subquery => `<li class="list-group-item bg-transparent">${subquery}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // Standard format (either WebRAgent or web search)
                const icon = isWebSearch ? 'bi-globe' : 'bi-chat-dots';
                
                html = `
                    <h4 class="h5 fw-bold mb-2"><i class="bi ${icon} me-2"></i>Generated Response:</h4>
                    <div class="badge-container mb-3">
                        ${badgeHTML}
                    </div>
                    <div class="llm-response mb-4">
                        ${data.response}
                    </div>
                `;
            }
            
            return html;
        },
        
        setupInteractions: function() {
            // This method can be used for setting up other interactive elements
            // Document preview links are already handled in displayResults
        },
        
        buildSourcesHTML: function(contexts, searchType) {
            let html = '';
            
            contexts.forEach((context, index) => {
                // Check if this is web search or document search
                const isWebResult = searchType === 'web' || context.source_type === 'web' || context.url;
                
                const pageInfo = !isWebResult && context.page_number ? 
                    `<span class="context-page">Page ${context.page_number}</span>` : '';
                
                if (isWebResult) {
                    // Web search result
                    html += `
                        <div class="source-document card mb-3">
                            <div class="card-header py-2 px-3 bg-light">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong class="d-flex align-items-center">
                                            <i class="bi bi-globe me-2"></i>
                                            <a href="${context.url || '#'}" target="_blank" class="external-link">
                                                ${context.document_title}
                                            </a>
                                        </strong>
                                        <small class="text-muted">${context.url ? new URL(context.url).hostname : 'Web source'}</small>
                                    </div>
                                    <span class="badge bg-primary rounded-pill">Score: ${context.score.toFixed(2)}</span>
                                </div>
                            </div>
                            <div class="card-body py-2 px-3">
                                <p class="mb-0 source-content">${context.content}</p>
                                <div class="mt-2">
                                    <a href="${context.url || '#'}" target="_blank" class="btn btn-sm btn-outline-primary">
                                        <i class="bi bi-box-arrow-up-right me-1"></i>Open Source
                                    </a>
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    // Document search result
                    html += `
                        <div class="source-document card mb-3">
                            <div class="card-header py-2 px-3 bg-light">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong class="d-flex align-items-center">
                                            <i class="bi bi-file-earmark-text me-2"></i>
                                            <a href="javascript:void(0)" class="context-link" 
                                               data-document-id="${context.document_id}" 
                                               data-preview-url="/document/${context.document_id}/preview">
                                                ${context.document_title}
                                            </a>
                                        </strong>
                                        ${pageInfo}
                                    </div>
                                    <span class="badge bg-primary rounded-pill">Score: ${context.score.toFixed(2)}</span>
                                </div>
                            </div>
                            <div class="card-body py-2 px-3">
                                <p class="mb-0 source-content">${context.content}</p>
                            </div>
                        </div>
                    `;
                }
            });
            
            return html;
        },
        
        handleError: function(error) {
            this.hideLoading();
            
            // Hide empty placeholder
            const emptyPlaceholder = document.getElementById('empty-results-placeholder');
            if (emptyPlaceholder) {
                emptyPlaceholder.style.display = 'none';
            }
            
            // Hide sources container
            const sourcesContainer = document.getElementById('sources-container');
            if (sourcesContainer) {
                sourcesContainer.style.display = 'none';
            }
            
            // Show error in response container
            const responseContainer = document.querySelector(config.selectors.responseContainer);
            if (responseContainer) {
                responseContainer.style.display = 'block';
                responseContainer.innerHTML = `
                    <div class="alert alert-danger enhanced-alert">
                        <i class="bi bi-exclamation-triangle-fill me-2 alert-icon"></i>
                        <div class="alert-message">
                            <strong>Error processing your request</strong>
                            <p class="mb-0">Please try again or check your connection.</p>
                        </div>
                    </div>
                `;
            }
        }
    };
    
    /**
     * Mind map generation and handling
     */
    const MindmapHandler = {
        // HTML templates for different mindmap states
        templates: {
            loading: `
                <html>
                <head>
                    <style>
                        body {
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 400px;
                            margin: 0;
                            background-color: #f8f9fa;
                        }
                        .loading-container {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            text-align: center;
                        }
                        .spinner {
                            border: 4px solid rgba(0, 0, 0, 0.1);
                            border-left-color: #0d6efd;
                            border-radius: 50%;
                            width: 40px;
                            height: 40px;
                            animation: spin 1s linear infinite;
                            margin-bottom: 1rem;
                        }
                        @keyframes spin {
                            to { transform: rotate(360deg); }
                        }
                    </style>
                </head>
                <body>
                    <div class="loading-container">
                        <div class="spinner"></div>
                        <p>Generating mind map visualization...</p>
                    </div>
                </body>
                </html>
            `,
            
            error: `
                <html>
                <head>
                    <style>
                        body {
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 400px;
                            margin: 0;
                            padding: 1rem;
                            background-color: #f8f9fa;
                            color: #dc3545;
                            text-align: center;
                        }
                        .error-icon {
                            font-size: 3rem;
                            margin-bottom: 1rem;
                        }
                    </style>
                </head>
                <body>
                    <div>
                        <div class="error-icon">⚠️</div>
                        <h3>Error Generating Mind Map</h3>
                        <p>We couldn't create the visualization. Please try again.</p>
                    </div>
                </body>
                </html>
            `,
            
            placeholder: `
                <html>
                <head>
                    <style>
                        body {
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100%;
                            margin: 0;
                            background-color: #f8f9fa;
                            text-align: center;
                        }
                        .placeholder-container {
                            max-width: 80%;
                        }
                        .icon {
                            font-size: 3rem;
                            color: #6c757d;
                            margin-bottom: 1rem;
                        }
                    </style>
                </head>
                <body>
                    <div class="placeholder-container">
                        <div class="icon">🧠</div>
                        <h3>Mind Map Visualization</h3>
                        <p>Submit a query to generate a mind map visualization of the answer.</p>
                    </div>
                </body>
                </html>
            `
        },
        
        init: function() {
            // Set up close button handler
            const closeMindmapBtn = document.querySelector(config.selectors.closeMindmapBtn);
            if (closeMindmapBtn) {
                closeMindmapBtn.addEventListener('click', this.reset.bind(this));
            }
            
            // Initialize placeholder if needed
            this.setDefaultPlaceholder();
        },
        
        setDefaultPlaceholder: function() {
            const mindmapFrame = document.querySelector(config.selectors.mindmapFrame);
            if (mindmapFrame && mindmapFrame.src === 'about:blank') {
                mindmapFrame.srcdoc = this.templates.placeholder;
            }
        },
        
        reset: function() {
            const mindmapFrame = document.querySelector(config.selectors.mindmapFrame);
            if (mindmapFrame) {
                mindmapFrame.srcdoc = this.templates.placeholder;
            }
        },
        
        generate: function(query, contexts) {
            const mindmapContainer = document.querySelector(config.selectors.mindmapContainer);
            const iframe = document.querySelector(config.selectors.mindmapFrame);
            
            if (!mindmapContainer || !iframe) return;
            
            // Display loading state in iframe
            iframe.srcdoc = this.templates.loading;
            
            // Format context for the request
            let contextText = '';
            if (contexts && contexts.length > 0) {
                contextText = contexts.map(ctx => {
                    return `From ${ctx.document_title}: ${ctx.content}`;
                }).join('\n\n');
            }
            
            // Create form data
            const formData = new FormData();
            formData.append('query', query);
            formData.append('context', contextText);
            
            // Send request to generate mindmap
            fetch(config.endpoints.mindmap, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.text();
            })
            .then(html => {
                // Set the iframe content to the received HTML
                iframe.srcdoc = html;
            })
            .catch(error => {
                console.error('Error generating mindmap:', error);
                iframe.srcdoc = this.templates.error;
            });
        }
    };
    
    /**
     * Document preview handling
     */
    const DocumentPreview = {
        // HTML templates for different document preview states
        templates: {
            placeholder: `
                <html>
                <head>
                    <style>
                        body {
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100%;
                            margin: 0;
                            background-color: #f8f9fa;
                            text-align: center;
                        }
                        .placeholder-container {
                            max-width: 80%;
                        }
                        .icon {
                            font-size: 3rem;
                            color: #6c757d;
                            margin-bottom: 1rem;
                        }
                    </style>
                </head>
                <body>
                    <div class="placeholder-container">
                        <div class="icon">📄</div>
                        <h3>Document Preview</h3>
                        <p>Select a source document from the search results to view its content here.</p>
                    </div>
                </body>
                </html>
            `
        },
        
        init: function() {
            // Initialize Bootstrap modal if available
            const modalElement = document.querySelector(config.selectors.documentPreviewModal);
            if (modalElement && typeof bootstrap !== 'undefined') {
                modal = new bootstrap.Modal(modalElement);
                
                // Setup modal events
                modalElement.addEventListener('hidden.bs.modal', this.handleModalClose);
                modalElement.addEventListener('show.bs.modal', this.handleModalShow);
            }
            
            // Set up document preview pane handling
            this.setupPreviewPane();
            
            // Initial setup for any context links already in the page
            this.setupLinks();
            
            // Set default placeholder for document preview
            this.setDefaultPlaceholder();
        },
        
        setupLinks: function() {
            document.querySelectorAll(config.selectors.contextLinks).forEach(link => {
                link.addEventListener('click', this.handleLinkClick.bind(this));
            });
        },
        
        setupPreviewPane: function() {
            // Close preview button for inline preview pane
            const closePreviewBtn = document.getElementById('close-preview');
            if (closePreviewBtn) {
                closePreviewBtn.addEventListener('click', function() {
                    const previewIframe = document.getElementById('preview-iframe');
                    const previewTitle = document.getElementById('preview-title');
                    
                    if (previewIframe) {
                        // Show placeholder instead of blank page when closing
                        previewIframe.srcdoc = DocumentPreview.templates.placeholder;
                    }
                    if (previewTitle) previewTitle.innerHTML = '<i class="bi bi-file-earmark-text me-2"></i>Document Preview';
                });
            }
        },
        
        setDefaultPlaceholder: function() {
            // Set a placeholder in the document preview iframe if empty
            const previewIframe = document.getElementById('preview-iframe');
            if (previewIframe && previewIframe.src === 'about:blank') {
                previewIframe.srcdoc = this.templates.placeholder;
            }
        },
        
        handleLinkClick: function(e) {
            const link = e.currentTarget || this;
            const documentId = link.getAttribute('data-document-id');
            const previewUrl = link.getAttribute('data-preview-url');
            const documentTitle = link.textContent.trim();
            
            // Check if we should use modal or inline preview
            const useModal = document.querySelector(config.selectors.documentPreviewModal) !== null;
            
            if (useModal) {
                // Update modal content
                const modalTitleElement = document.querySelector(config.selectors.modalPreviewTitle);
                const modalIframeElement = document.querySelector(config.selectors.modalPreviewIframe);
                const modalDownloadElement = document.querySelector(config.selectors.modalPreviewDownload);
                
                if (modalTitleElement) modalTitleElement.textContent = documentTitle;
                if (modalIframeElement) modalIframeElement.src = previewUrl;
                if (modalDownloadElement) {
                    const rawUrl = config.endpoints.documentRaw.replace('%ID%', documentId);
                    modalDownloadElement.href = rawUrl;
                }
                
                // Show the modal
                if (modal) modal.show();
            } else {
                // Use inline preview pane
                const previewTitle = document.getElementById('preview-title');
                const previewIframe = document.getElementById('preview-iframe');
                const previewDownload = document.getElementById('preview-download');
                
                if (previewTitle) previewTitle.innerHTML = '<i class="bi bi-file-earmark-text me-2"></i>' + documentTitle;
                
                // The key change: Set the iframe src directly to the preview URL, not srcdoc
                if (previewIframe) {
                    // This is the important fix - use src not srcdoc for external content
                    previewIframe.src = previewUrl;
                    
                    // Remove any srcdoc content that might be conflicting
                    previewIframe.removeAttribute('srcdoc');
                }
                
                if (previewDownload) {
                    previewDownload.href = `/document/${documentId}/raw`;
                }
            }
        },
        
        handleModalShow: function() {
            // Add blur effect to modal backdrop
            setTimeout(() => {
                const backdrop = document.querySelector('.modal-backdrop');
                if (backdrop) backdrop.classList.add('blur-backdrop');
            }, 10);
        },
        
        handleModalClose: function() {
            // Clean up when modal is closed
            const modalIframeElement = document.querySelector(config.selectors.modalPreviewIframe);
            if (modalIframeElement) {
                // Show placeholder instead of blank page
                if (DocumentPreview && DocumentPreview.templates) {
                    modalIframeElement.srcdoc = DocumentPreview.templates.placeholder;
                } else {
                    modalIframeElement.src = 'about:blank';
                }
            }
            
            // Remove blur effect
            const backdrop = document.querySelector('.modal-backdrop');
            if (backdrop) backdrop.classList.remove('blur-backdrop');
        }
    };
    
    /**
     * Admin functionality handlers
     */
    const AdminHandlers = {
        init: function() {
            // Confirmation for collection deletion
            document.querySelectorAll(config.selectors.deleteCollectionForms).forEach(form => {
                form.addEventListener('submit', this.confirmDelete.bind(this, 'collection'));
            });
            
            // Confirmation for document deletion
            document.querySelectorAll(config.selectors.deleteDocumentForms).forEach(form => {
                form.addEventListener('submit', this.confirmDelete.bind(this, 'document'));
            });
        },
        
        confirmDelete: function(itemType, e) {
            const message = `Are you sure you want to delete this ${itemType}? This action cannot be undone.`;
            if (!confirm(message)) {
                e.preventDefault();
            }
        }
    };
    
    /**
     * Utility functions for common tasks
     */
    const Utilities = {
        // Add observer to watch for DOM changes and apply handlers to new elements
        setupDynamicContentObserver: function() {
            // Set up MutationObserver to handle dynamically added content
            const responseContainer = document.querySelector(config.selectors.responseContainer);
            if (!responseContainer) return;
            
            const observer = new MutationObserver(mutations => {
                mutations.forEach(mutation => {
                    if (mutation.addedNodes.length) {
                        // When new content is added, set up interactions
                        DocumentPreview.setupLinks();
                    }
                });
            });
            
            // Start observing the container
            observer.observe(responseContainer, { 
                childList: true,
                subtree: true 
            });
        }
    };
    
    /**
     * Public API
     */
    return {
        init: function() {
            // Load user preferences
            Preferences.load();
            
            // Setup listeners for preference changes
            const generateMindmapCheckbox = document.querySelector(config.selectors.generateMindmap);
            if (generateMindmapCheckbox) {
                generateMindmapCheckbox.addEventListener('change', Preferences.save);
            }
            
            // Set up source type selection (new radio buttons)
            const useCollectionOption = document.getElementById('use-collection-option');
            const useWebOption = document.getElementById('use-web-option');
            const collectionSelectorContainer = document.getElementById('collection-selector-container');
            const webSearchOptionsContainer = document.getElementById('web-search-options-container');
            
            // Helper function to update submit button text
            const updateSubmitButtonText = function() {
                const submitBtn = document.querySelector('#query-form button[type="submit"]');
                if (!submitBtn) return;
                
                const isWebSearch = document.getElementById('use-web-option').checked;
                const isDeepSearch = document.getElementById('use-deep-search').checked;
                const isAgentSearch = document.getElementById('use-agent-search').checked;
                
                if (isWebSearch) {
                    if (isDeepSearch) {
                        if (isAgentSearch) {
                            submitBtn.innerHTML = '<span class="spinner-submit"><i class="bi bi-robot me-1"></i><i class="bi bi-search-heart me-1"></i>Enhanced Deep Web Search</span>';
                        } else {
                            submitBtn.innerHTML = '<span class="spinner-submit"><i class="bi bi-search-heart me-1"></i>Deep Web Search</span>';
                        }
                    } else {
                        if (isAgentSearch) {
                            submitBtn.innerHTML = '<span class="spinner-submit"><i class="bi bi-robot me-1"></i><i class="bi bi-globe me-1"></i>Enhanced Web Search</span>';
                        } else {
                            submitBtn.innerHTML = '<span class="spinner-submit"><i class="bi bi-globe me-1"></i>Web Search</span>';
                        }
                    }
                } else {
                    // Collection search
                    if (isAgentSearch) {
                        submitBtn.innerHTML = '<span class="spinner-submit"><i class="bi bi-robot me-1"></i>Enhanced Collection Search</span>';
                    } else {
                        submitBtn.innerHTML = '<span class="spinner-submit"><i class="bi bi-send me-1"></i>Collection Search</span>';
                    }
                }
            };
            
            // Setup collection/web toggle handlers
            if (useCollectionOption && useWebOption) {
                // Collection option selected
                useCollectionOption.addEventListener('change', function() {
                    if (this.checked) {
                        // Show collection selector, hide web options
                        if (collectionSelectorContainer) collectionSelectorContainer.classList.remove('d-none');
                        if (webSearchOptionsContainer) webSearchOptionsContainer.classList.add('d-none');
                        
                        // Enable collection select
                        const collectionSelect = document.querySelector(config.selectors.collectionSelect);
                        if (collectionSelect) {
                            collectionSelect.disabled = false;
                            collectionSelect.setAttribute('aria-required', 'true');
                            collectionSelect.setAttribute('required', '');
                        }
                        
                        // Update hidden fields for form submission compatibility
                        const useWebSearch = document.getElementById('use-web-search');
                        if (useWebSearch) useWebSearch.checked = false;
                        
                        // Update submit button text
                        updateSubmitButtonText();
                        
                        // Save preference
                        sessionStorage.setItem('source_type_main', 'collection');
                    }
                });
                
                // Web option selected
                useWebOption.addEventListener('change', function() {
                    if (this.checked) {
                        // Show web options, hide collection selector
                        if (collectionSelectorContainer) collectionSelectorContainer.classList.add('d-none');
                        if (webSearchOptionsContainer) webSearchOptionsContainer.classList.remove('d-none');
                        
                        // Disable collection select since it's not needed
                        const collectionSelect = document.querySelector(config.selectors.collectionSelect);
                        if (collectionSelect) {
                            collectionSelect.disabled = true;
                            collectionSelect.setAttribute('aria-required', 'false');
                            collectionSelect.removeAttribute('required');
                        }
                        
                        // Ensure web search is enabled (now a hidden checkbox)
                        const useWebSearch = document.getElementById('use-web-search');
                        if (useWebSearch) useWebSearch.checked = true;
                        
                        // Update submit button text
                        updateSubmitButtonText();
                        
                        // Save preference
                        sessionStorage.setItem('source_type_main', 'web');
                    }
                });
                
                // Load saved preference
                const savedSourceType = sessionStorage.getItem('source_type_main');
                if (savedSourceType === 'web') {
                    useWebOption.checked = true;
                    // Trigger change event to update UI
                    const event = new Event('change');
                    useWebOption.dispatchEvent(event);
                } else {
                    useCollectionOption.checked = true;
                    // Trigger change event to update UI
                    const event = new Event('change');
                    useCollectionOption.dispatchEvent(event);
                }
            }
            
            // Deep search toggle
            const useDeepSearchCheckbox = document.querySelector(config.selectors.useDeepSearch);
            if (useDeepSearchCheckbox) {
                useDeepSearchCheckbox.addEventListener('change', function() {
                    Preferences.save();
                    updateSubmitButtonText();
                });
            }
            
            // Web search is now handled through the radio buttons, so no direct event handler needed
            
            // Agent search handler
            const useAgentSearchCheckbox = document.querySelector(config.selectors.useAgentSearch);
            if (useAgentSearchCheckbox) {
                useAgentSearchCheckbox.addEventListener('change', function() {
                    Preferences.save();
                    
                    // Get the strategy dropdown container
                    const strategyContainer = document.querySelector(config.selectors.agentStrategyContainer);
                    
                    // Update UI based on checkbox state
                    if (this.checked) {
                        // Show the strategy dropdown
                        if (strategyContainer) {
                            strategyContainer.classList.remove('d-none');
                        }
                    } else {
                        // Hide the strategy dropdown
                        if (strategyContainer) {
                            strategyContainer.classList.add('d-none');
                        }
                    }
                    
                    // Update submit button text
                    updateSubmitButtonText();
                });
            }
            
            // Agent strategy dropdown
            const agentStrategySelect = document.querySelector(config.selectors.agentStrategy);
            if (agentStrategySelect) {
                agentStrategySelect.addEventListener('change', function() {
                    Preferences.save();
                });
            }
            
            const maxResultsInput = document.querySelector(config.selectors.maxResults);
            if (maxResultsInput) {
                maxResultsInput.addEventListener('change', Preferences.save);
            }
            
            // Initialize components
            QueryProcessor.init();
            DocumentPreview.init();
            AdminHandlers.init();
            
            // Initialize tooltips if Bootstrap is available
            if (typeof bootstrap !== 'undefined') {
                var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
                var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                    return new bootstrap.Tooltip(tooltipTriggerEl);
                });
                
                // Initialize collapse for search options
                const searchOptionsToggle = document.querySelector('[data-bs-target="#searchOptionsCollapse"]');
                if (searchOptionsToggle) {
                    searchOptionsToggle.addEventListener('click', function() {
                        const icon = this.querySelector('i');
                        if (icon) {
                            if (icon.classList.contains('bi-chevron-up')) {
                                icon.classList.replace('bi-chevron-up', 'bi-chevron-down');
                            } else {
                                icon.classList.replace('bi-chevron-down', 'bi-chevron-up');
                            }
                        }
                    });
                    
                    // Get the collapse element
                    const searchOptionsCollapse = document.getElementById('searchOptionsCollapse');
                    if (searchOptionsCollapse) {
                        searchOptionsCollapse.addEventListener('hidden.bs.collapse', function() {
                            const icon = searchOptionsToggle.querySelector('i');
                            if (icon) icon.classList.replace('bi-chevron-up', 'bi-chevron-down');
                        });
                        
                        searchOptionsCollapse.addEventListener('shown.bs.collapse', function() {
                            const icon = searchOptionsToggle.querySelector('i');
                            if (icon) icon.classList.replace('bi-chevron-down', 'bi-chevron-up');
                        });
                    }
                }
            }
            
            // Setup observer for dynamic content
            Utilities.setupDynamicContentObserver();
            
            // Initialize mindmap handler
            MindmapHandler.init();
        }
    };
})();

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', WebRAgentApp.init);