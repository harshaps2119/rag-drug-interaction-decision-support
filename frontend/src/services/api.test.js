import { describe, it, expect, vi, afterEach } from "vitest";
import { searchDrugs, checkInteraction, checkMultipleInteractions, checkHealth, ApiError } from "./api";

function mockFetchOnce(status, body, ok = status >= 200 && status < 300) {
  global.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
  });
}

function mockFetchNetworkFailure() {
  global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
}

function mockFetchMalformedJson(status = 200, ok = true) {
  global.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => {
      throw new SyntaxError("Unexpected token");
    },
  });
}

describe("services/api", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("searchDrugs", () => {
    it("returns [] without calling fetch for a query under 2 characters", async () => {
      global.fetch = vi.fn();
      const results = await searchDrugs("a");
      expect(results).toEqual([]);
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it("returns [] for an empty/whitespace query", async () => {
      global.fetch = vi.fn();
      expect(await searchDrugs("")).toEqual([]);
      expect(await searchDrugs("   ")).toEqual([]);
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it("calls the search endpoint and returns results for a valid query", async () => {
      mockFetchOnce(200, [{ rxcui: "11289", normalized_name: "warfarin", term_type: "IN" }]);
      const results = await searchDrugs("warf");
      expect(results).toHaveLength(1);
      expect(results[0].normalized_name).toBe("warfarin");
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/api/drugs/search?q=warf"),
        expect.any(Object)
      );
    });
  });

  describe("checkInteraction", () => {
    it("posts the two drug names and returns the parsed response", async () => {
      mockFetchOnce(200, { interaction_assessment: "pair_specific_evidence_found" });
      const result = await checkInteraction("warfarin", "ibuprofen");
      expect(result.interaction_assessment).toBe("pair_specific_evidence_found");

      const [, options] = global.fetch.mock.calls[0];
      expect(options.method).toBe("POST");
      expect(JSON.parse(options.body)).toEqual({ drug_a: "warfarin", drug_b: "ibuprofen" });
    });

    it("throws ApiError with code network_error when fetch itself fails", async () => {
      mockFetchNetworkFailure();
      await expect(checkInteraction("warfarin", "ibuprofen")).rejects.toMatchObject({
        name: "ApiError",
        code: "network_error",
      });
    });

    it("throws ApiError with the backend's error code/message on a 422", async () => {
      mockFetchOnce(422, { error: { code: "validation_error", message: "drug_a: too long", request_id: "r1" } }, false);
      await expect(checkInteraction("x".repeat(500), "ibuprofen")).rejects.toMatchObject({
        code: "validation_error",
        message: "drug_a: too long",
        status: 422,
        requestId: "r1",
      });
    });

    it("throws ApiError with code rate_limit_exceeded on a 429", async () => {
      mockFetchOnce(429, { error: { code: "rate_limit_exceeded", message: "Rate limit exceeded. Try again in 30 seconds.", request_id: "r2" } }, false);
      await expect(checkInteraction("warfarin", "ibuprofen")).rejects.toMatchObject({
        code: "rate_limit_exceeded",
        status: 429,
      });
    });

    it("throws a safe generic ApiError on a 500 with no error body", async () => {
      mockFetchOnce(500, {}, false);
      await expect(checkInteraction("warfarin", "ibuprofen")).rejects.toMatchObject({
        status: 500,
      });
    });

    it("throws ApiError with code malformed_response when the response body isn't valid JSON", async () => {
      mockFetchMalformedJson(200, true);
      await expect(checkInteraction("warfarin", "ibuprofen")).rejects.toMatchObject({
        code: "malformed_response",
      });
    });
  });

  describe("checkMultipleInteractions", () => {
    it("posts the drug list", async () => {
      mockFetchOnce(200, { pairs: [], total_pairs: 0, request_id: "r3" });
      await checkMultipleInteractions(["warfarin", "ibuprofen", "aspirin"]);
      const [, options] = global.fetch.mock.calls[0];
      expect(JSON.parse(options.body)).toEqual({ drugs: ["warfarin", "ibuprofen", "aspirin"] });
    });
  });

  describe("checkHealth", () => {
    it("calls GET /api/health", async () => {
      mockFetchOnce(200, { status: "ok", checks: {}, request_id: "r4" });
      await checkHealth();
      expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/api/health"), expect.any(Object));
    });
  });

  it("ApiError is a real Error subclass with the expected fields", () => {
    const err = new ApiError("oops", { code: "x", status: 400, requestId: "r5" });
    expect(err).toBeInstanceOf(Error);
    expect(err.message).toBe("oops");
    expect(err.code).toBe("x");
    expect(err.status).toBe(400);
    expect(err.requestId).toBe("r5");
  });
});
