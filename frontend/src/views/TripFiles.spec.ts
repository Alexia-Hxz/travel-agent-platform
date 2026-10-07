import { flushPromises, mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import TripFiles from './TripFiles.vue'

describe('TripFiles', () => {
  it('lists a trip file returned by the API', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify([
            {
              id: 31,
              trip_id: 7,
              original_filename: 'transport-ticket.pdf',
              stored_filename: 'abc.pdf',
              content_type: 'application/pdf',
              file_size: 1024,
              file_path: 'data/uploads/7/abc.pdf',
              parse_status: 'pending',
              parsed_text: null,
              parsed_result: null,
              parse_error: null,
              parsed_at: null,
              created_at: '2026-09-05T10:00:00',
            },
          ]),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )

    const wrapper = mount(TripFiles, {
      props: { tripId: 7 },
      global: { plugins: [ElementPlus] },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('transport-ticket.pdf')
    expect(wrapper.text()).toContain('Parse file')
  })

  it('offers multimodal analysis for an uploaded image', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify([
            {
              id: 32, trip_id: 7, original_filename: 'museum.jpg', stored_filename: 'museum.jpg', content_type: 'image/jpeg',
              file_size: 1024, file_path: 'data/uploads/7/museum.jpg', parse_status: 'pending', parsed_text: null,
              parsed_result: null, parse_error: null, parsed_at: null, created_at: '2026-09-05T10:00:00',
            },
          ]),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )

    const wrapper = mount(TripFiles, { props: { tripId: 7 }, global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.text()).toContain('Analyze with agent')
  })

  it('notifies the parent after saving an analyzed place', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn()
        .mockResolvedValueOnce(new Response(JSON.stringify([{
          id: 32, trip_id: 7, original_filename: 'museum.jpg', stored_filename: 'museum.jpg', content_type: 'image/jpeg',
          file_size: 1024, file_path: 'data/uploads/7/museum.jpg', parse_status: 'pending', parsed_text: null,
          parsed_result: null, parse_error: null, parsed_at: null, created_at: '2026-09-05T10:00:00',
        }]), { status: 200 }))
        .mockResolvedValueOnce(new Response(JSON.stringify({
          summary: 'Detected place', document_type: 'ticket', reference_number: null,
          raw_text: 'Museum', place: { name: 'Museum', type: 'attraction', address: 'Road', latitude: null, longitude: null, image_url: null, rating: null, price_level: null, opening_hours: null, source_url: null },
          itinerary: null,
        }), { status: 200 }))
        .mockResolvedValueOnce(new Response(JSON.stringify({ place: { id: 99 }, itinerary: null }), { status: 201 })),
    )
    const wrapper = mount(TripFiles, { props: { tripId: 7 }, global: { plugins: [ElementPlus] } })
    await flushPromises()
    const analyzeButton = wrapper.findAll('button').find((button) => button.text().includes('Analyze with agent'))
    expect(analyzeButton).toBeDefined()
    await analyzeButton!.trigger('click')
    await flushPromises()
    const confirmButton = wrapper.findAll('button').find((button) => button.text().includes('Confirm save'))
    expect(confirmButton).toBeDefined()
    await confirmButton!.trigger('click')
    await flushPromises()

    expect(wrapper.emitted('places-updated')).toHaveLength(1)
  })
})
