use std::collections::HashMap;

use egui::Context;
use llmux::{LlmConfig, Provider};
use llmux_keystore::ProviderStore;

/// Retained-mode configuration widget. Renders a button that opens a modal
/// panel. Owns all draft state internally. Changes are committed only on **Save**.
pub struct LlmConfigWidget {
    open: bool,
    drafts: HashMap<Provider, LlmConfig>,
    selected: Provider,
}

impl LlmConfigWidget {
    /// Construct the widget. Loads credentials from `ProviderStore` immediately
    /// so they are ready when the modal is first opened.
    pub fn new(config: Option<&LlmConfig>) -> Self {
        let drafts = load_drafts();
        Self {
            open: false,
            drafts,
            selected: config
                .map(|v| v.provider.clone())
                .unwrap_or(Provider::default()),
        }
    }

    /// Returns the currently saved config for the selected provider, if any.
    pub fn config(&self) -> Option<&LlmConfig> {
        self.drafts.get(&self.selected)
    }

    /// Call every frame inside `eframe::App::update()`.
    ///
    /// Renders a **⚙ LLM Settings** button. When clicked, opens a modal panel.
    /// Returns `Some(LlmConfig)` when the user clicks **Save**, `None` otherwise.
    pub fn show(&mut self, ctx: &Context, ui: &mut egui::Ui) -> Option<LlmConfig> {
        if ui.button("⚙ LLM Settings").clicked() {
            self.open = true;
        }

        if !self.open {
            return None;
        }

        let screen = ctx.input(|i| {
            i.viewport().outer_rect.unwrap_or(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(1280.0, 800.0),
            ))
        });
        let width = screen.width() * 0.8;
        let height = screen.height() * 0.8;

        let mut result: Option<LlmConfig> = None;
        let mut close = false;

        egui::Window::new("LLM Settings")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .fixed_size([width, height])
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                // Provider tabs
                ui.horizontal(|ui| {
                    for provider in all_providers() {
                        let selected = self.selected == provider;
                        if ui.selectable_label(selected, provider.label()).clicked() {
                            self.selected = provider;
                        }
                    }
                });

                ui.separator();

                if let Some(draft) = self.drafts.get_mut(&self.selected) {
                    egui::Grid::new("llm_config_fields")
                        .num_columns(2)
                        .spacing([12.0, 8.0])
                        .show(ui, |ui| {
                            if self.selected.needs_key() {
                                ui.label("API Key");
                                ui.add(
                                    egui::TextEdit::singleline(&mut draft.api_key)
                                        .password(false)
                                        .desired_width(f32::INFINITY),
                                );
                                ui.end_row();
                            }

                            if matches!(self.selected, Provider::Ollama) {
                                ui.label("Base URL");
                                let mut url = draft.base_url.clone().unwrap_or_default();
                                ui.add(
                                    egui::TextEdit::singleline(&mut url)
                                        .desired_width(f32::INFINITY),
                                );
                                draft.base_url = if url.is_empty() { None } else { Some(url) };
                                ui.end_row();
                            }

                            ui.label("Model");
                            ui.add(
                                egui::TextEdit::singleline(&mut draft.model)
                                    .desired_width(f32::INFINITY),
                            );
                            ui.end_row();
                        });
                }

                ui.add_space(8.0);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::BOTTOM), |ui| {
                    if ui.button("Save").clicked() {
                        if let Ok(mut store) = ProviderStore::load() {
                            for (provider, config) in &self.drafts {
                                store.set(provider.clone(), config.clone());
                            }
                            let _ = store.save();
                        }
                        result = self.drafts.get(&self.selected).cloned();
                        close = true;
                    }

                    if ui.button("Cancel").clicked() {
                        // Reload drafts from store, discarding unsaved edits
                        self.drafts = load_drafts();
                        close = true;
                    }
                });
            });

        if close {
            self.open = false;
        }

        result
    }
}

fn load_drafts() -> HashMap<Provider, LlmConfig> {
    let store = ProviderStore::load().unwrap_or_else(|_| ProviderStore {
        configs: HashMap::new(),
    });

    all_providers()
        .into_iter()
        .map(|provider| {
            let config = store.get(&provider).cloned().unwrap_or_else(|| LlmConfig {
                model: provider.default_model().into(),
                api_key: String::new(),
                base_url: if matches!(provider, Provider::Ollama) {
                    Some("http://localhost:11434".into())
                } else {
                    None
                },
                provider: provider.clone(),
            });
            (provider, config)
        })
        .collect()
}

fn all_providers() -> [Provider; 4] {
    [
        Provider::Anthropic,
        Provider::Gemini,
        Provider::OpenAI,
        Provider::Ollama,
    ]
}
